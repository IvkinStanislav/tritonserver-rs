use std::time::Duration;

use anyhow::Result;
use env_logger::builder as log_builder;
use log::{info, LevelFilter};
use tritonserver_rs::{memory::Sample, options::Options, response::Output, Request, Response, Server};

pub use utils::*;

pub const DEFAULT_BACKEND_PATH: &str = "/opt/tritonserver/backends";

pub async fn run_pipeline<P: Pipeline>(mut pipeline: P) -> Result<()> {
    log_builder().filter_level(LevelFilter::Info).init();

    #[cfg(feature = "gpu")]
    tritonserver_rs::init_cuda()?;

    let mut opts = Options::new(pipeline.model_repo())?;

    opts.exit_timeout(Duration::from_secs(5))?
        .backend_directory(pipeline.backends_path())?;

    #[cfg(not(feature = "gpu"))]
    opts.pinned_memory_pool_byte_size(0)?;

    let server = Server::new(opts).await?;

    let mut request = server.create_request(pipeline.model_name(), -1)?;
    request.add_default_allocator();
    pipeline.add_inputs(&mut request)?;

    let fut = request.infer_async()?;
    info!("Request successfully start");

    let response = fut.await?;
    pipeline.parse_result(response)?;

    server.stop()?;
    info!("Server successfully stop");

    Ok(())
}

pub trait Pipeline {
    fn model_repo(&self) -> String;

    fn backends_path(&self) -> String;

    fn model_name(&self) -> String;

    fn add_inputs(&mut self, request: &mut Request) -> Result<()>;

    fn parse_result(&mut self, result: Response) -> Result<()>;
}

pub mod utils {
    use super::*;

    pub fn output_as_ref<T: Sample>(out: &Output) -> Vec<T> {
        let buf = out.get_buffer();
        let buf = buf.get_owned_slice(..).unwrap();
        let (pref, res, suf) = unsafe { buf.align_to::<T>() };
        if !pref.is_empty() && !suf.is_empty() {
            log::error!("Error parsing values of {} output", out.name);
        }
        res.to_vec()
    }

    /// Transpose [A, B] to [B, A]. B comes as last_dim arg.
    pub fn transpose<V: AsRef<[T]>, T: Copy>(source: V, last_dim: usize) -> Vec<T> {
        let chunks_num = source.as_ref().len() / last_dim;
        let mut res = Vec::new();
        for j in 0..last_dim {
            for i in 0..chunks_num {
                res.push(source.as_ref()[i * last_dim + j]);
            }
        }
        res
    }
}
