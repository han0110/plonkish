use std::time::Instant;

#[allow(dead_code)]
pub struct TimerInfo {
    pub msg: String,
    pub time: Instant,
    ended: bool,
}

#[cfg(not(feature = "timer"))]
pub use disabled::{end_timer, start_timer};
#[cfg(feature = "timer")]
pub use enabled::{end_timer, start_timer};

#[cfg(feature = "timer")]
mod enabled {
    use crate::util::timer::{Instant, TimerInfo};
    use colored::Colorize;
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    static COUNT: AtomicUsize = AtomicUsize::new(0);
    const MDOT: &str = "·";

    pub fn start_timer<T: ToString>(msg: impl Fn() -> T) -> TimerInfo {
        let indent = MDOT.repeat(2 * COUNT.load(Ordering::Relaxed));
        let prefix = "Start:".yellow().bold();
        let msg = msg().to_string();
        println!("{}{:8} {}", indent, prefix, msg);
        COUNT.fetch_add(1, Ordering::Relaxed);
        TimerInfo {
            msg,
            time: Instant::now(),
            ended: false,
        }
    }

    pub fn end_timer(mut info: TimerInfo) {
        print_end_msg(info.time.elapsed(), &info.msg);
        info.ended = true;
    }

    impl Drop for TimerInfo {
        fn drop(&mut self) {
            if !self.ended {
                print_end_msg(self.time.elapsed(), &self.msg);
            }
        }
    }

    fn print_end_msg(duration: Duration, msg: &str) {
        let duration = {
            let secs = duration.as_secs();
            let millis = duration.subsec_millis();
            let micros = duration.subsec_micros() % 1000;
            let nanos = duration.subsec_nanos() % 1000;
            if secs != 0 {
                format!("{}.{:03}s", secs, millis)
            } else if millis > 0 {
                format!("{}.{:03}ms", millis, micros)
            } else if micros > 0 {
                format!("{}.{:03}µs", micros, nanos)
            } else {
                format!("{}ns", nanos)
            }
        }
        .bold();
        COUNT.fetch_sub(1, Ordering::Relaxed);
        let indent_len = 2 * COUNT.load(Ordering::Relaxed);
        let indent = MDOT.repeat(indent_len);
        let prefix = "End:".green().bold();
        let msg = format!("{} ", msg);
        println!(
            "{}{:8} {:.<pad$}{}",
            indent,
            prefix,
            msg,
            duration,
            pad = 75 - indent_len
        );
    }
}

#[cfg(not(feature = "timer"))]
mod disabled {
    use crate::util::timer::{Instant, TimerInfo};

    pub fn start_timer<T: ToString>(_: impl Fn() -> T) -> TimerInfo {
        TimerInfo {
            msg: String::new(),
            time: Instant::now(),
            ended: false,
        }
    }

    pub fn end_timer(_: TimerInfo) {}
}
