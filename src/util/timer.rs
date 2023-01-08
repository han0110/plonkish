use ark_std::{end_timer, perf_trace::TimerInfo as InnerTimerInfo, start_timer};

#[allow(dead_code)]
pub struct TimerInfo {
    pub inner: InnerTimerInfo,
    ended: bool,
}

pub fn start_timer<T: ToString>(_msg: impl Fn() -> T) -> TimerInfo {
    TimerInfo {
        inner: start_timer!(|| _msg().to_string()),
        ended: false,
    }
}

pub fn end_timer(mut info: TimerInfo) {
    end_timer!(info.inner);
    info.ended = true;
}

impl Drop for TimerInfo {
    fn drop(&mut self) {
        if !self.ended {
            end_timer!(self.inner);
        }
    }
}
