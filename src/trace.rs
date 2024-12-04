use std::{
    ffi::{c_void, CStr},
    mem::{forget, transmute},
    os::raw::c_char,
    ptr::{null, null_mut},
    sync::Arc,
    time::Duration,
};

use crate::{
    error::{Error, CSTR_CONVERT_ERROR_PLUG},
    sys,
};

/// Trace levels
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Level {
    Disabled = sys::tritonserver_tracelevel_enum_TRITONSERVER_TRACE_LEVEL_DISABLED,
    Min = sys::tritonserver_tracelevel_enum_TRITONSERVER_TRACE_LEVEL_MIN,
    Max = sys::tritonserver_tracelevel_enum_TRITONSERVER_TRACE_LEVEL_MAX,
}

impl Level {
    /// Get the string representation of a trace level.
    pub fn as_str(self) -> &'static str {
        unsafe {
            let ptr = sys::TRITONSERVER_InferenceTraceLevelString(self as u32);
            assert!(!ptr.is_null());
            CStr::from_ptr(ptr)
                .to_str()
                .unwrap_or(CSTR_CONVERT_ERROR_PLUG)
        }
    }
}

/// Enum representation of inference status.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Activity {
    RequestStart = sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_REQUEST_START,
    QueueStart = sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_QUEUE_START,
    ComputeStart = sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_COMPUTE_START,
    ComputeInputEnd = sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_COMPUTE_INPUT_END,
    ComputeOutputStart =
        sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_COMPUTE_OUTPUT_START,
    ComputeEnd = sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_COMPUTE_END,
    RequestEnd = sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_REQUEST_END,
}

/// Inference event handler trait.
pub trait TraceHandler: Send + 'static {
    /// This function is invoked each time the `event` accures.
    ///
    /// `trace`: Trace object that was reported.
    /// Note that child traces of constructed one also are reported with this fn.
    /// Check [Trace::new_with_handle] for more info.\
    /// `event`: activity that has occurred. \
    /// `event_time`: time when event occured.
    fn trace_activity(&self, trace: &Trace, event: Activity, event_time: Duration);
}

/// Inference object that provides custom tracing.
///
/// Is constructed with [TraceHandler] object that is activated each time an event occures.
pub struct Trace {
    pub(crate) ptr: *mut sys::TRITONSERVER_InferenceTrace,
}

impl Trace {
    /// Create a new inference trace object.
    ///
    /// The handler.trace_activity() will be called to report activity
    /// for this trace as well as for any child traces that are spawned
    /// by this one, and so the trace_activity should check the trace object (first argument)
    /// to determine specifically what activity is being reported.
    ///
    /// `level`: The tracing level. \
    /// `parent_id`: The parent trace id for this trace.
    /// A value of 0 indicates that there is not parent trace. \
    /// `handle`: The callback function where activity for the trace is reported.
    pub fn new_with_handle<H: TraceHandler>(
        level: Level,
        parent_id: u64,
        handle: Arc<H>,
    ) -> Result<Self, Error> {
        let mut ptr = null_mut::<sys::TRITONSERVER_InferenceTrace>();
        let raw_handle = Arc::into_raw(handle.clone());

        match triton_call!(sys::TRITONSERVER_InferenceTraceNew(
            &mut ptr as *mut _,
            level as u32,
            parent_id,
            Some(activity_wraper::<H>),
            Some(delete::<H>),
            raw_handle as *mut c_void,
        )) {
            Ok(_) => {
                assert!(!ptr.is_null());
                Ok(Trace { ptr })
            }
            Err(err) => {
                unsafe {
                    Arc::from_raw(raw_handle);
                }
                Err(err)
            }
        }
    }

    /// Get the id associated with the trace.
    /// Every trace is assigned an id that is unique across all traces created for a Triton server.
    pub fn get_id(&self) -> Result<u64, Error> {
        let mut id: u64 = 0;
        triton_call!(
            sys::TRITONSERVER_InferenceTraceId(self.ptr, &mut id as *mut _),
            id
        )
    }

    /// Get the parent id associated with the trace. \
    /// The parent id indicates a parent-child relationship between two traces.
    /// A parent id value of 0 indicates that there is no parent trace.
    pub fn get_parent_id(&self) -> Result<u64, Error> {
        let mut id: u64 = 0;
        triton_call!(
            sys::TRITONSERVER_InferenceTraceParentId(self.ptr, &mut id as *mut _),
            id
        )
    }

    /// Get the name of the model associated with the trace.
    pub fn get_model_name(&self) -> Result<&str, Error> {
        let mut ptr = null::<c_char>();
        triton_call!(sys::TRITONSERVER_InferenceTraceModelName(
            self.ptr,
            &mut ptr as *mut _
        ))?;

        assert!(!ptr.is_null());
        Ok(unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .unwrap_or(CSTR_CONVERT_ERROR_PLUG))
    }

    /// Get the version of the model associated with the trace.
    pub fn get_model_version(&self) -> Result<i64, Error> {
        let mut version: i64 = 0;
        triton_call!(
            sys::TRITONSERVER_InferenceTraceModelVersion(self.ptr, &mut version as *mut _),
            version
        )
    }
}

// It normally never being called.
impl Drop for Trace {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                sys::TRITONSERVER_InferenceTraceDelete(self.ptr);
            }
        }
    }
}

unsafe extern "C" fn delete<H: TraceHandler>(
    this: *mut sys::TRITONSERVER_InferenceTrace,
    userp: *mut c_void,
) {
    sys::TRITONSERVER_InferenceTraceDelete(this);
    if !userp.is_null() {
        Arc::from_raw(userp as *mut H);
    }
}

unsafe extern "C" fn activity_wraper<H: TraceHandler>(
    trace: *mut sys::TRITONSERVER_InferenceTrace,
    activity: sys::TRITONSERVER_InferenceTraceActivity,
    timestamp_ns: u64,
    userp: *mut ::std::os::raw::c_void,
) {
    if !userp.is_null() {
        let handle = Arc::from_raw(userp as *mut H);

        let trace = Trace { ptr: trace };

        let activity: Activity = transmute(activity);
        let timestamp = Duration::from_nanos(timestamp_ns);

        handle.trace_activity(&trace, activity, timestamp);

        // Drop will be in delete method.
        forget(handle);
        forget(trace);
    }
}
