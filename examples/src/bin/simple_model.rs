use anyhow::Result;
use config_manager::{config, ConfigInit};
use tritonserver_rs::{Buffer, Request, Response};

use triton_examples::*;

#[tokio::main]
async fn main() -> Result<()> {
    let pipe = SimplePipe::parse()?;
    println!("{pipe:?}");

    run_pipeline(pipe).await
}

#[derive(Debug)]
#[config]
struct SimplePipe {
    #[source(clap(long, short), default = "DEFAULT_BACKEND_PATH.into()")]
    backend: String,
    #[source(clap(long, short), default = "\"models/simple_model\".into()")]
    model_path: String,
    #[source(clap(long, short), default = "vec![1.0,2.0,3.0,4.0,5.0,6.0]")]
    input_vector: Vec<f32>,
}

impl Pipeline for SimplePipe {
    fn backends_path(&self) -> String {
        self.backend.clone()
    }

    fn model_repo(&self) -> String {
        self.model_path.clone()
    }

    fn model_name(&self) -> String {
        "adder".into()
    }

    fn add_inputs(&mut self, request: &mut Request) -> Result<()> {
        request.add_input_with_dims(
            "input",
            Buffer::from(&self.input_vector),
            [1, 1, 1, self.input_vector.len() as _],
        )?;
        Ok(())
    }

    fn parse_result(&mut self, result: Response) -> Result<()> {
        let out = result.get_output("output").unwrap();
        let result: f32 = out.get_buffer().as_ref()[0];
        log::info!("Sum by dims of {:?} is: {result}", self.input_vector);
        Ok(())
    }
}
