use gnuplot::{AutoOption, AxesCommon, Figure, PlotOption};
use itertools::Itertools;
use std::{
    collections::{BTreeMap, HashMap},
    env::args,
    fmt::{self, Display},
    fs::{create_dir, File},
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Command, Stdio},
    time::Duration,
};

const OUTPUT_DIR: &str = "./target/bench";

fn main() {
    let (verbose, logs) = parse_args();
    let logs_by_system = Log::parse(&logs)
        .into_iter()
        .filter(|log| log.name.contains("prove"))
        .group_by(|log| log.name.clone())
        .into_iter()
        .fold(
            vec![BTreeMap::<_, _>::new(); System::iter().count()],
            |mut by_system, (name, logs)| {
                let (name, k) = name.split_once('-').unwrap();
                let system = System::iter()
                    .find(|system| name.starts_with(format!("{system}").as_str()))
                    .unwrap();
                let mut logs = logs.collect_vec();
                logs.iter_mut().for_each(|log| log.group(&system.key_fn()));
                by_system[system as usize].insert(k.parse::<usize>().unwrap(), Log::average(&logs));
                by_system
            },
        );
    if verbose {
        for (system, logs) in System::iter().zip(logs_by_system.iter()) {
            for (k, log) in logs.iter() {
                println!("{system}-{k}");
                println!("{log}");
            }
        }
    }

    let cost_breakdowns_by_system = System::iter()
        .zip(logs_by_system.iter())
        .map(|(system, logs)| {
            logs.iter()
                .map(|(k, log)| (*k, system.cost_breakdown(log)))
                .collect::<BTreeMap<_, _>>()
        })
        .collect_vec();
    for (system, cost_breakdowns) in System::iter().zip(cost_breakdowns_by_system.iter()) {
        plot_cost_breakdown(system, cost_breakdowns);
    }
    plot_comparison(&cost_breakdowns_by_system);
}

fn parse_args() -> (bool, Vec<String>) {
    let (verbose, logs) = args().chain(["".to_string()]).tuple_windows().fold(
        (false, None),
        |(mut verbose, mut logs), (key, value)| {
            match key.as_str() {
                "-" => {
                    logs = Some(
                        std::io::stdin()
                            .lines()
                            .try_collect::<_, Vec<_>, _>()
                            .unwrap(),
                    )
                }
                "--log" => {
                    logs = Some(
                        BufReader::new(File::open(value).unwrap())
                            .lines()
                            .try_collect::<_, Vec<_>, _>()
                            .unwrap(),
                    )
                }
                "--verbose" => {
                    verbose = true;
                }
                _ => {}
            };
            (verbose, logs)
        },
    );
    (
        verbose,
        logs.expect("Either \"--log <LOG>\" or \"-\" specified"),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum System {
    HyperPlonk,
    UniHyperPlonk,
    Halo2,
    EspressoHyperPlonk,
}

impl System {
    fn iter() -> impl Iterator<Item = System> {
        [
            System::HyperPlonk,
            System::UniHyperPlonk,
            System::Halo2,
            System::EspressoHyperPlonk,
        ]
        .into_iter()
    }

    fn key_fn(&self) -> impl Fn(&Log) -> (bool, &str) + '_ {
        move |log| match self {
            System::HyperPlonk | System::UniHyperPlonk | System::Halo2 => (
                false,
                log.name.split([' ', '-']).next().unwrap_or(&log.name),
            ),
            System::EspressoHyperPlonk => {
                let is_multiexp = log.name.starts_with("msm of size")
                    || matches!(log.name.as_str(), "commit witnesses" | "commit");
                (
                    is_multiexp,
                    if is_multiexp {
                        "multiexp"
                    } else if log.name.ends_with("-th round") {
                        "rounds"
                    } else if log.name.ends_with("-th round eval") {
                        "rounds eval"
                    } else {
                        &log.name
                    },
                )
            }
        }
    }

    fn cost_breakdown(&self, log: &Log) -> Vec<(&'static str, Duration)> {
        let costs = match self {
            System::HyperPlonk => vec![
                (
                    "all",
                    vec![
                        vec!["variable_base_msm"],
                        vec!["sum_check_prove"],
                        vec!["evals"],
                        vec!["pcs_batch_open"],
                    ],
                    None,
                ),
                ("multiexp", vec![vec!["variable_base_msm"]], None),
                ("sum check", vec![vec!["sum_check_prove"]], None),
                (
                    "pcs multiexp",
                    vec![vec!["pcs_batch_open", "variable_base_msm"]],
                    None,
                ),
                (
                    "pcs sum check",
                    vec![vec!["pcs_batch_open", "sum_check_prove"]],
                    None,
                ),
                (
                    "pcs rest",
                    vec![vec!["evals"], vec!["pcs_batch_open"]],
                    Some(vec![
                        vec!["pcs_batch_open", "variable_base_msm"],
                        vec!["pcs_batch_open", "sum_check_prove"],
                    ]),
                ),
            ],
            System::UniHyperPlonk => vec![
                (
                    "all",
                    vec![
                        vec!["variable_base_msm"],
                        vec!["sum_check_prove"],
                        vec!["prove_multilinear_eval"],
                    ],
                    None,
                ),
                ("multiexp", vec![vec!["variable_base_msm"]], None),
                ("sum check", vec![vec!["sum_check_prove"]], None),
                (
                    "mleval multiexp",
                    vec![
                        vec!["prove_multilinear_eval", "variable_base_msm"],
                        vec![
                            "prove_multilinear_eval",
                            "pcs_batch_open",
                            "variable_base_msm",
                        ],
                    ],
                    None,
                ),
                (
                    "mleval fft",
                    vec![vec!["prove_multilinear_eval", "fft"]],
                    None,
                ),
                (
                    "mleval rest",
                    vec![vec!["prove_multilinear_eval"]],
                    Some(vec![
                        vec!["prove_multilinear_eval", "variable_base_msm"],
                        vec![
                            "prove_multilinear_eval",
                            "pcs_batch_open",
                            "variable_base_msm",
                        ],
                        vec!["prove_multilinear_eval", "fft"],
                    ]),
                ),
            ],
            System::Halo2 => vec![
                (
                    "all",
                    vec![
                        vec!["ifft"],
                        vec!["variable_base_msm"],
                        vec!["fft"],
                        vec!["quotient_polys"],
                        vec!["evals"],
                        vec!["pcs_batch_open"],
                    ],
                    None,
                ),
                ("fft", vec![vec!["ifft"], vec!["fft"]], None),
                ("multiexp", vec![vec!["variable_base_msm"]], None),
                (
                    "pcs multiexp",
                    vec![vec!["pcs_batch_open", "variable_base_msm"]],
                    None,
                ),
                (
                    "pcs rest",
                    vec![vec!["evals"], vec!["pcs_batch_open"]],
                    Some(vec![vec!["pcs_batch_open", "variable_base_msm"]]),
                ),
                ("quotient", vec![vec!["quotient_polys"]], None),
            ],
            System::EspressoHyperPlonk => vec![
                (
                    "all",
                    vec![
                        vec!["hyperplonk proving", "multiexp"],
                        vec![
                            "hyperplonk proving",
                            "Permutation check on w_i(x)",
                            "Permutation check prove",
                            "prod_check prove",
                            "multiexp",
                        ],
                        vec!["hyperplonk proving", "ZeroCheck on f"],
                        vec![
                            "hyperplonk proving",
                            "Permutation check on w_i(x)",
                            "Permutation check prove",
                            "prod_check prove",
                            "zerocheck in product check",
                        ],
                        vec!["hyperplonk proving", "opening and evaluations"],
                        vec!["hyperplonk proving", "deferred batch openings prod(x)"],
                    ],
                    None,
                ),
                (
                    "multiexp",
                    vec![
                        vec!["hyperplonk proving", "multiexp"],
                        vec![
                            "hyperplonk proving",
                            "Permutation check on w_i(x)",
                            "Permutation check prove",
                            "prod_check prove",
                            "multiexp",
                        ],
                    ],
                    None,
                ),
                (
                    "sum check",
                    vec![
                        vec!["hyperplonk proving", "ZeroCheck on f"],
                        vec![
                            "hyperplonk proving",
                            "Permutation check on w_i(x)",
                            "Permutation check prove",
                            "prod_check prove",
                            "zerocheck in product check",
                        ],
                    ],
                    None,
                ),
                (
                    "pcs multiexp",
                    vec![vec![
                        "hyperplonk proving",
                        "deferred batch openings prod(x)",
                        "multi open ",
                        "pcs open",
                        "open mle with ",
                        "rounds",
                        "multiexp",
                    ]],
                    None,
                ),
                (
                    "pcs sum check",
                    vec![vec![
                        "hyperplonk proving",
                        "deferred batch openings prod(x)",
                        "multi open ",
                        "sum check prove",
                    ]],
                    None,
                ),
                (
                    "pcs rest",
                    vec![
                        vec!["hyperplonk proving", "opening and evaluations"],
                        vec!["hyperplonk proving", "deferred batch openings prod(x)"],
                    ],
                    Some(vec![
                        vec![
                            "hyperplonk proving",
                            "deferred batch openings prod(x)",
                            "multi open ",
                            "pcs open",
                            "open mle with ",
                            "rounds",
                            "multiexp",
                        ],
                        vec![
                            "hyperplonk proving",
                            "deferred batch openings prod(x)",
                            "multi open ",
                            "sum check prove",
                        ],
                    ]),
                ),
            ],
        }
        .into_iter()
        .map(|(name, adds, subs)| {
            let add = adds
                .iter()
                .map(|indices| log.duration(indices))
                .sum::<Duration>();
            let sub = subs
                .map(|subs| subs.iter().map(|indices| log.duration(indices)).sum())
                .unwrap_or_default();
            (name, add - sub)
        })
        .collect_vec();
        assert_eq!(
            costs[0].1,
            costs[1..].iter().map(|(_, duration)| duration).sum(),
        );
        costs
    }
}

impl Display for System {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            System::HyperPlonk => write!(f, "hyperplonk"),
            System::UniHyperPlonk => write!(f, "unihyperplonk"),
            System::Halo2 => write!(f, "halo2"),
            System::EspressoHyperPlonk => write!(f, "espresso_hyperplonk"),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Log {
    name: String,
    depth: usize,
    duration: Duration,
    components: Vec<Log>,
}

impl Log {
    fn parse(logs: &[String]) -> Vec<Log> {
        let mut stack = Vec::new();
        logs.iter().fold(Vec::new(), |mut logs, log| {
            let (indent, log) = log.rsplit_once('·').unwrap_or(("", log));
            if log.len() < 9 {
                return logs;
            }

            let (prefix, log) = log.split_at(9);
            let depth = (indent.len() + 2) / 4;
            if depth == stack.len() && prefix.starts_with("Start:") {
                stack.push(Log {
                    name: log.to_string(),
                    depth,
                    ..Default::default()
                });
            } else if prefix.starts_with("End:") {
                let name = log.rsplit_once(' ').unwrap().0;
                let duration = log
                    .rsplit_once("..")
                    .map(|(_, duration)| parse_duration(duration))
                    .unwrap_or_default();
                if let Some(idx) = stack.iter().rposition(|log| log.name == name) {
                    stack.truncate(idx + 1);
                    let mut log = stack.pop().unwrap();
                    log.duration = duration;
                    if let Some(parent) = stack.last_mut() {
                        parent.components.push(log);
                    } else {
                        logs.push(log);
                    }
                }
            }
            logs
        })
    }

    fn average<'a>(logs: impl IntoIterator<Item = &'a Log>) -> Log {
        let logs = logs.into_iter().collect_vec();
        let mut avg = logs[0].clone();
        assert!(!logs.iter().any(|log| log.name != avg.name));

        avg.duration = logs.iter().map(|log| log.duration).sum::<Duration>() / logs.len() as u32;
        avg.components = (0..avg.components.len())
            .map(|idx| Log::average(logs.iter().map(|log| &log.components[idx]).collect_vec()))
            .collect();
        avg
    }

    fn group(&mut self, key_fn: &impl Fn(&Log) -> (bool, &str)) {
        let mut grouped =
            self.components
                .iter_mut()
                .fold(HashMap::<_, Log>::new(), |mut grouped, log| {
                    let (is_unit, key) = key_fn(log);
                    grouped
                        .entry(key.to_string())
                        .and_modify(|group| {
                            group.duration += log.duration;
                            for (lhs, rhs) in group.components.iter_mut().zip(log.components.iter())
                            {
                                assert!(lhs.components.is_empty() && rhs.components.is_empty());
                                lhs.duration += rhs.duration;
                            }
                        })
                        .or_insert_with(|| {
                            let mut group = log.clone();
                            if is_unit {
                                group.components.truncate(0);
                            }
                            group
                        });
                    grouped
                });
        self.name = key_fn(self).1.to_string();
        self.components.retain_mut(|log| {
            grouped
                .remove(key_fn(log).1)
                .map(|mut group| {
                    group.group(key_fn);
                    *log = group;
                })
                .is_some()
        });
    }

    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
        max_depth: usize,
        percentage: Option<Percentage<Duration>>,
    ) -> fmt::Result {
        if self.depth > max_depth {
            return Ok(());
        }

        writeln!(
            f,
            "{prefix}{name:<padding$}{duration:?}{percentage}",
            prefix = "·".repeat(2 * self.depth),
            name = self.name,
            padding = 40,
            duration = self.duration,
            percentage = percentage
                .map(|percentage| format!(" ({percentage})"))
                .unwrap_or_default()
        )?;

        let rest_duration = self
            .duration
            .checked_sub(self.components.iter().map(|log| log.duration).sum())
            .unwrap_or_default();
        let rest =
            (!self.components.is_empty() && rest_duration >= Duration::from_millis(1)).then(|| {
                Log {
                    name: "rest".to_string(),
                    depth: self.depth + 1,
                    duration: rest_duration,
                    ..Default::default()
                }
            });
        for log in self.components.iter().chain(&rest) {
            let percentage = Some(Percentage::new(log.duration, self.duration));
            log.fmt(f, max_depth, percentage)?;
        }

        Ok(())
    }

    fn duration(&self, names: &[&'static str]) -> Duration {
        let mut log = self;
        for name in names.iter() {
            log = log
                .components
                .iter()
                .find(|log| log.name.starts_with(name))
                .unwrap();
        }
        log.duration
    }
}

impl Display for Log {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt(f, usize::MAX, None)
    }
}

struct Percentage<T> {
    numer: T,
    denom: T,
}

impl<T> Percentage<T> {
    fn new(numer: T, denom: T) -> Self {
        Self { numer, denom }
    }
}

impl Display for Percentage<Duration> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let percentage = 100.0 * self.numer.as_nanos() as f64 / self.denom.as_nanos() as f64;
        write!(f, "{percentage:.2}%")
    }
}

fn parse_duration(duration: &str) -> Duration {
    let mid = duration.rfind(|a: char| a.is_ascii_digit()).unwrap() + 1;
    let (value, unit) = duration.split_at(mid);
    let base = match unit {
        "ns" => 1f64,
        "µs" => 1_000f64,
        "ms" => 1_000_000f64,
        "s" => 1_000_000_000f64,
        _ => unreachable!(),
    };
    Duration::from_nanos((value.parse::<f64>().unwrap() * base) as u64)
}

fn plot_cost_breakdown(system: System, cost_breakdowns: &BTreeMap<usize, Vec<(&str, Duration)>>) {
    if cost_breakdowns.is_empty() {
        return;
    }

    if cost_breakdowns.len() > 1 {
        let x = cost_breakdowns.keys().copied().collect_vec();
        let lines = (0..cost_breakdowns.first_key_value().unwrap().1.len())
            .map(|idx| {
                (
                    cost_breakdowns.first_key_value().unwrap().1[idx].0,
                    cost_breakdowns
                        .values()
                        .map(|cost_breakdown| cost_breakdown[idx].1.as_nanos() as f64 / 1_000_000.0)
                        .collect_vec(),
                )
            })
            .collect_vec();
        let mut fg = Figure::new();
        let axes = fg
            .set_enhanced_text(false)
            .set_pre_commands("set key top left")
            .axes2d();
        axes.set_title(format!("{system} cost breakdown").as_str(), &[])
            .set_x_label("log #constraints", &[])
            .set_y_label("time (millisecond)", &[])
            .set_x_grid(true)
            .set_y_grid(true)
            .set_x_ticks(Some((AutoOption::Fix(1.0), 0)), &[], &[])
            .set_y_log(Some(10.0))
            .set_x_range(AutoOption::Fix(10.0), AutoOption::Fix(25.0))
            .set_y_range(AutoOption::Fix(0.1), AutoOption::Fix(1000000.0));
        for (name, values) in lines.iter() {
            axes.lines_points(x.clone(), values, &[PlotOption::Caption(name)]);
        }
        save(fg, format!("cost_breakdown_{system}.png"));
    }

    let (k, cost_breakdown) = cost_breakdowns.last_key_value().unwrap();
    let num_components = cost_breakdown.len() - 1;
    let y_range_start = -(1.0 + 0.25 * num_components as f64);
    let legend_x = (cost_breakdown[1..]
        .iter()
        .map(|(name, _)| name.len())
        .max()
        .unwrap()
        + 7) as f64
        * -0.035;
    let sum = cost_breakdown[0].1.as_nanos() as u64;
    let data = cost_breakdown[1..]
        .iter()
        .map(|(name, duration)| format!("\"{name}\" {}", duration.as_nanos()))
        .join("\n");
    let input = format!(
        r#"
            set terminal png font "monaco"
            set output "{OUTPUT_DIR}/cost_breakdown_pie_{k}_{system}.png"

            unset key
            unset tics
            unset border

            set size ratio -1
            set xrange [-1:1]
            set yrange [{y_range_start}:1]
            set style fill solid 1 noborder

            sum = {sum}
            angle = 0
            i = 0
            j = 0

            plot '-' using (0):(0):(1):(angle):(angle=angle+360*$2/sum):(i=i+1) with circles lc var ,\
                '-' using ({legend_x}):(-1-(j=j+1)*0.25):(sprintf('%05.2f%% %s', ($2)*100/sum, stringcolumn(1))) with labels left offset 3,0 ,\
                for [k=1:{num_components}] '+' using ({legend_x}):(-1-k*0.25) with points pt 5 ps 4 lc k
            {data}
            e
            {data}
            e
        "#,
    );
    gnuplot(input);
}

fn plot_comparison(cost_breakdowns_by_system: &[BTreeMap<usize, Vec<(&str, Duration)>>]) {
    if cost_breakdowns_by_system[System::HyperPlonk as usize].is_empty() {
        return;
    }

    let (min, max) = cost_breakdowns_by_system
        .iter()
        .filter_map(|cost_breakdowns| cost_breakdowns.keys().copied().minmax().into_option())
        .reduce(|(min, max), (first, last)| (min.min(first), max.max(last)))
        .unwrap();
    let x = (min..=max).collect_vec();
    let hyperplonk_cost_breakdowns = &cost_breakdowns_by_system[System::HyperPlonk as usize];
    let lines = System::iter()
        .zip(cost_breakdowns_by_system.iter())
        .skip(1)
        .filter(|(_, cost_breakdowns)| !cost_breakdowns.is_empty())
        .map(|(system, cost_breakdowns)| {
            let [numer, denom] =
                [cost_breakdowns, hyperplonk_cost_breakdowns].map(|cost_breakdowns| {
                    x.iter()
                        .map(|k| cost_breakdowns[k][0].1.as_nanos() as f64)
                        .collect_vec()
                });
            let ratio = numer
                .iter()
                .zip(denom.iter())
                .map(|(numer, denom)| numer / denom)
                .collect_vec();
            (format!("{system}/{}", System::HyperPlonk), ratio)
        })
        .collect_vec();

    if lines.is_empty() {
        return;
    }

    let mut fg = Figure::new();
    let axes = fg
        .set_enhanced_text(false)
        .set_pre_commands("set key top left")
        .axes2d();
    axes.set_title("comparison", &[])
        .set_x_label("log #constraints", &[])
        .set_y_label("time ratio", &[])
        .set_x_grid(true)
        .set_y_grid(true)
        .set_x_ticks(Some((AutoOption::Fix(1.0), 0)), &[], &[])
        .set_y_ticks(Some((AutoOption::Fix(0.5), 0)), &[], &[])
        .set_x_range(AutoOption::Fix(10.0), AutoOption::Fix(24.0))
        .set_y_range(AutoOption::Fix(0.0), AutoOption::Fix(5.0));
    for (name, values) in lines {
        axes.lines_points(x.clone(), values, &[PlotOption::Caption(&name)]);
    }
    save(fg, "comparison.png");
}

fn save(mut fg: Figure, name: impl AsRef<str>) {
    if !Path::new(OUTPUT_DIR).exists() {
        create_dir(OUTPUT_DIR).unwrap();
    }
    let path = format!("{OUTPUT_DIR}/{}", name.as_ref());
    fg.set_terminal("png font \"monaco\"", path.as_ref());
    fg.show().unwrap().wait().unwrap();
    fg.close();
}

fn gnuplot(input: impl AsRef<str>) {
    let mut gnuplot = Command::new("gnuplot")
        .arg("-p")
        .stdin(Stdio::piped())
        .spawn()
        .expect("to have gnuplot installed");
    gnuplot
        .stdin
        .as_mut()
        .expect("to have stdin")
        .write_all(input.as_ref().as_bytes())
        .unwrap();
    gnuplot.wait().unwrap();
}
