use std::{ffi::CString, os::unix::prelude::OsStrExt, path::Path, ptr::null_mut, time::Duration};

use bitflags::bitflags;

use crate::{
    error::{Error, ErrorCode},
    sys, to_cstring,
};

bitflags! {
    /// Triton server control model modes.
    pub struct Control: u32 {
        /// The models in model repository will be loaded on startup. \
        /// After startup any changes to the model repository will be ignored. \
        /// Calling Server::poll_model_repository will result in an error.
        const NONE = sys::tritonserver_modelcontrolmode_enum_TRITONSERVER_MODEL_CONTROL_NONE;
        /// The models in model repository will be loaded on startup. \
        /// The model repository can be polled periodically using Server::poll_model_repository and the server will load, \
        /// unload, and updated models according to changes in the model repository.
        const POLL = sys::tritonserver_modelcontrolmode_enum_TRITONSERVER_MODEL_CONTROL_POLL;
        /// The models in model repository will not be loaded on startup. \
        /// The corresponding model control APIs must be called to load / unload a model in the model repository.
        const EXPLICIT = sys::tritonserver_modelcontrolmode_enum_TRITONSERVER_MODEL_CONTROL_EXPLICIT;
    }

}

bitflags! {
    /// Triton server rate limit modes.
    pub struct Limit: u32 {
        // The rate limiting is turned off and the inference gets executed whenever an instance is available.
        const OFF = sys::tritonserver_ratelimitmode_enum_TRITONSERVER_RATE_LIMIT_OFF;
        /// The rate limiting prioritizes the inference execution using the number of times each instance has got a chance to run. \
        /// The execution gets to run only when its resource constraints are satisfied.
        const EXEC_COUNT = sys::tritonserver_ratelimitmode_enum_TRITONSERVER_RATE_LIMIT_EXEC_COUNT;
    }
}

/// Triton server creation options.
#[derive(Debug)]
pub struct Options(pub(crate) *mut sys::TRITONSERVER_ServerOptions);

impl Options {
    /// Create a new server options object. \
    /// The path must be the full absolute path to the model repository. \
    /// This function can be called multiple times with different paths to set multiple model repositories. \
    /// Note that if a model is not unique across all model repositories at any time, the model will not be available.
    pub fn new<P: AsRef<Path>>(repository: P) -> Result<Self, Error> {
        let path = repository
            .as_ref()
            .canonicalize()
            .map_err(|err| Error::new(ErrorCode::InvalidArg, err.to_string()))
            .and_then(|path| {
                CString::new(path.as_os_str().as_bytes())
                    .map_err(|err| Error::new(ErrorCode::InvalidArg, err.to_string()))
            })?;
        let mut this = null_mut::<sys::TRITONSERVER_ServerOptions>();

        triton_call!(sys::TRITONSERVER_ServerOptionsNew(&mut this as *mut _))?;

        assert!(!this.is_null());
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetModelRepositoryPath(
                this,
                path.as_bytes().as_ptr() as *const i8,
            ),
            Self(this)
        )
    }

    /// Set the textual ID for the server in a server options. The ID is a name that identifies the server.
    pub fn server_id<I: AsRef<str>>(&mut self, id: I) -> Result<&mut Self, Error> {
        let id = to_cstring(id)?;
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetServerId(self.0, id.as_ptr()),
            self
        )
    }

    /// Set the model to be loaded at startup in a server options. \
    /// The model must be present in one, and only one, of the specified model repositories. \
    /// This function can be called multiple times with different model name to set multiple startup models. \
    /// Note that it only takes affect with [Control::EXPLICIT] set.
    pub fn startup_model<S: AsRef<str>>(&mut self, model: S) -> Result<&mut Self, Error> {
        let model = to_cstring(model)?;
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetStartupModel(self.0, model.as_ptr()),
            self
        )
    }

    /// Set the model control mode in a server options. For each mode the models will be managed as the following:
    ///
    /// [Control::NONE]: the models in model repository will be loaded on startup.
    /// After startup any changes to the model repository will be ignored. Calling [poll_model_repository](crate::Server::poll_model_repository) will result in an error.
    ///
    /// [Control::POLL]: the models in model repository will be loaded on startup.
    /// The model repository can be polled periodically using [poll_model_repository](crate::Server::poll_model_repository) and the server will load,
    /// unload, and updated models according to changes in the model repository.
    ///
    /// [Control::EXPLICIT]: the models in model repository will not be loaded on startup.
    /// The corresponding model control APIs must be called to load / unload a model in the model repository.
    pub fn model_control_mode(&mut self, mode: Control) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetModelControlMode(self.0, mode.bits()),
            self
        )
    }

    /// Enable or disable strict model configuration handling in a server options.
    pub fn strict_model_config(&mut self, enable: bool) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetStrictModelConfig(self.0, enable),
            self
        )
    }

    /// Set the rate limit mode.
    ///
    /// [Limit::EXEC_COUNT]: The rate limiting prioritizes the inference execution using the number of times each instance has got a chance to run.
    /// The execution gets to run only when its resource constraints are satisfied.
    ///
    /// [Limit::OFF]: The rate limiting is turned off and the inference gets executed whenever an instance is available.
    ///
    /// By default, execution count is used to determine the priorities.
    pub fn rate_limiter_mode(&mut self, mode: Limit) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetRateLimiterMode(self.0, mode.bits()),
            self
        )
    }

    /// Add resource count for rate limiting. \
    /// `name`: The name of the resource. \
    /// `count`: The count of the resource. \
    /// `device`: The device identifier for the resource.
    /// A value of -1 indicates that the specified number of resources are available on every device.
    ///
    /// The device value is ignored for a global resource. \
    /// The server will use the rate limiter configuration specified for instance groups in model config to determine whether resource is global. \
    /// In case of conflicting resource type in different model configurations, server will raise an appropriate error while loading model.
    pub fn add_rate_limiter_resource<N: AsRef<str>>(
        &mut self,
        name: N,
        count: u64,
        device: i32,
    ) -> Result<&mut Self, Error> {
        let name = to_cstring(name)?;
        triton_call!(
            sys::TRITONSERVER_ServerOptionsAddRateLimiterResource(
                self.0,
                name.as_ptr(),
                count as usize,
                device,
            ),
            self
        )
    }

    /// Set the total pinned memory byte size that the server can allocate . \
    /// The pinned memory pool will be shared across Triton itself and the backends that use MemoryManager to allocate memory. \
    /// `size`: The pinned memory pool byte size.
    pub fn pinned_memory_pool_byte_size(&mut self, size: u64) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(self.0, size),
            self
        )
    }

    /// Set the total CUDA memory byte size that the server can allocate on given GPU device. \
    /// The pinned memory pool will be shared across Triton itself and the backends that use MemoryManager to allocate memory. \
    /// `device`: The GPU device to allocate the memory pool. \
    /// `size`: The pinned memory pool byte size.
    pub fn cuda_memory_pool_byte_size(
        &mut self,
        device: i32,
        size: u64,
    ) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize(self.0, device, size),
            self
        )
    }

    /// Set the total response cache byte size that the server can allocate in CPU memory. \
    /// The response cache will be shared across all inference requests and across all models. \
    /// `size`: The total response cache byte size.
    pub fn response_cache_byte_size(&mut self, size: u64) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetResponseCacheByteSize(self.0, size),
            self
        )
    }

    /// Set the minimum support CUDA compute capability. \
    /// `capability`: The minimum CUDA compute capability.
    pub fn min_supported_compute_capability(
        &mut self,
        capability: f64,
    ) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(self.0, capability),
            self
        )
    }

    /// Enable or disable exit-on-error. True to enable exiting on initialization error, false to continue.
    pub fn exit_on_error(&mut self, enable: bool) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetExitOnError(self.0, enable),
            self
        )
    }

    /// Enable or disable strict readiness handling.
    pub fn strict_readiness(&mut self, enable: bool) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetStrictReadiness(self.0, enable),
            self
        )
    }

    /// Set the exit timeout.
    pub fn exit_timeout(&mut self, timeout: Duration) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetExitTimeout(self.0, timeout.as_secs().max(1) as _),
            self
        )
    }

    /// Set the number of threads used in buffer manager.
    pub fn buffer_manager_thread_count(&mut self, thread: usize) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetBufferManagerThreadCount(self.0, thread as _),
            self
        )
    }

    /// Enable or disable info level logging.
    pub fn log_info(&mut self, enable: bool) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetLogInfo(self.0, enable),
            self
        )
    }

    /// Enable or disable warning level logging.
    pub fn log_warn(&mut self, enable: bool) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetLogWarn(self.0, enable),
            self
        )
    }

    /// Enable or disable error level logging.
    pub fn log_error(&mut self, enable: bool) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetLogError(self.0, enable),
            self
        )
    }

    /// Set verbose logging level. Level zero disables verbose logging.
    pub fn log_verbose(&mut self, level: i32) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetLogVerbose(self.0, level),
            self
        )
    }

    /// Enable or disable metrics collection in a server options.
    pub fn metrics(&mut self, enable: bool) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetMetrics(self.0, enable),
            self
        )
    }

    /// Enable or disable GPU metrics collection in a server options.
    /// GPU metrics are collected if both this option and [Options::metrics] are set.
    pub fn gpu_metrics(&mut self, enable: bool) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetGpuMetrics(self.0, enable),
            self
        )
    }

    /// Set the interval for metrics collection in a server options.
    /// This is 2000 milliseconds by default.
    pub fn metrics_interval(&mut self, interval: Duration) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetMetricsInterval(
                self.0,
                interval.as_millis().max(1) as _,
            ),
            self
        )
    }

    /// Set the directory containing backend shared libraries. \
    /// This directory is searched last after the version and model directory
    /// in the model repository when looking for the backend shared library for a model. \
    /// If the backend is named 'be' the directory searched is 'backend_dir'/be/libtriton_be.so.
    pub fn backend_directory<P: AsRef<Path>>(&mut self, path: P) -> Result<&mut Self, Error> {
        let path = path
            .as_ref()
            .canonicalize()
            .map_err(|err| Error::new(ErrorCode::InvalidArg, err.to_string()))
            .and_then(|path| {
                CString::new(path.as_os_str().as_bytes())
                    .map_err(|err| Error::new(ErrorCode::InvalidArg, err.to_string()))
            })?;
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetBackendDirectory(self.0, path.as_ptr()),
            self
        )
    }

    /// Set the directory containing repository agent shared libraries. \
    /// This directory is searched when looking for the repository agent shared library for a model. \
    /// If the backend is named 'ra' the directory searched is 'repoagent_dir'/ra/libtritonrepoagent_ra.so.
    pub fn repo_agent_directory<P: AsRef<Path>>(&mut self, path: P) -> Result<&mut Self, Error> {
        let path = CString::new(path.as_ref().as_os_str().as_bytes())
            .map_err(|err| Error::new(ErrorCode::InvalidArg, format!("{}", err)))?;
        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetRepoAgentDirectory(self.0, path.as_ptr()),
            self
        )
    }

    /// Set a configuration setting for a named backend in a server options. \
    /// `name`: The name of the backend. \
    /// `setting`: The name of the setting. \
    /// `value`: The setting value.
    pub fn backend_config<N, S, V>(
        &mut self,
        name: N,
        setting: S,
        value: V,
    ) -> Result<&mut Self, Error>
    where
        N: AsRef<str>,
        S: AsRef<str>,
        V: AsRef<str>,
    {
        let name = to_cstring(name)?;
        let setting = to_cstring(setting)?;
        let value = to_cstring(value)?;

        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetBackendConfig(
                self.0,
                name.as_ptr(),
                setting.as_ptr(),
                value.as_ptr(),
            ),
            self
        )
    }

    /// Set a host policy setting for a given policy name in a server options. \
    /// `name`: The name of the policy. \
    /// `setting`: The name of the setting. \
    /// `value`: The setting value.
    pub fn host_policy<N, S, V>(
        &mut self,
        name: N,
        setting: S,
        value: V,
    ) -> Result<&mut Self, Error>
    where
        N: AsRef<str>,
        S: AsRef<str>,
        V: AsRef<str>,
    {
        let name = to_cstring(name)?;
        let setting = to_cstring(setting)?;
        let value = to_cstring(value)?;

        triton_call!(
            sys::TRITONSERVER_ServerOptionsSetHostPolicy(
                self.0,
                name.as_ptr(),
                setting.as_ptr(),
                value.as_ptr(),
            ),
            self
        )
    }
}

unsafe impl Send for Options {}

impl Drop for Options {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                let _ = sys::TRITONSERVER_ServerOptionsDelete(self.0);
            }
        }
    }
}
