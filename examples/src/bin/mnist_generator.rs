use anyhow::Result;
use config_manager::{config, ConfigInit};
use image::{DynamicImage, GenericImage, GrayImage};
use rand::Rng;
use tritonserver_rs::{Buffer, Request, Response};

use triton_examples::*;

#[tokio::main]
async fn main() -> Result<()> {
    let pipe = HandwritePipe::parse()?;
    println!("{pipe:?}");

    run_pipeline(pipe).await
}

#[config]
#[derive(Debug)]
pub struct HandwritePipe {
    #[source(clap(long, short), default = "DEFAULT_BACKEND_PATH.into()")]
    backend: String,
    #[source(clap(long, short), default = "\"models/mnist_generator\".into()")]
    model_path: String,
    #[source(
        clap(long = "output", short),
        default = "\"data/output/mnist.png\".into()"
    )]
    output_image_path: String,
    #[source(clap(long, short), default = 5)]
    table_size: usize,
}

impl Pipeline for HandwritePipe {
    fn backends_path(&self) -> String {
        self.backend.clone()
    }

    fn model_repo(&self) -> String {
        self.model_path.clone()
    }

    fn model_name(&self) -> String {
        "generator".into()
    }

    fn add_inputs(&mut self, request: &mut Request) -> Result<()> {
        const INPUT_LATENT_DIM: usize = 100;
        let batch_size = self.table_size * self.table_size;

        let mut rng = rand::thread_rng();

        let noizes = (0..(INPUT_LATENT_DIM * batch_size))
            .map(|_| rng.gen_range(0.0..1.0f32))
            .collect::<Vec<_>>();

        let buf = Buffer::from(noizes.as_slice());
        request.add_input_with_dims(
            "input",
            buf,
            [batch_size as _, INPUT_LATENT_DIM as _, 1, 1],
        )?;
        Ok(())
    }

    fn parse_result(&mut self, result: Response) -> Result<()> {
        let table_size = self.table_size;

        let out = result.get_output("output").unwrap();
        let img_content = output_as_ref::<f32>(out);
        let w = out.shape[2] as usize;
        let h = out.shape[3] as usize;

        let byte_imgs = img_content
            .iter()
            .cloned()
            .map(|pixel| (pixel * 255.0) as u8)
            .collect::<Vec<_>>();

        let images = byte_imgs.chunks(w * h).map(|byte_content| {
            DynamicImage::ImageLuma8(
                GrayImage::from_raw(w as _, h as _, byte_content.to_vec()).unwrap(),
            )
        });

        let mut compilation = DynamicImage::new_luma8((w * table_size) as _, (h * table_size) as _);
        images.enumerate().for_each(|(pos, small_img)| {
            let pos_x = pos % table_size;
            let pos_y = pos / table_size;
            compilation
                .copy_from(&small_img, (pos_x * w) as _, (pos_y * h) as _)
                .unwrap();
        });

        compilation.save(&self.output_image_path)?;

        log::info!("Image saved to: {:#?}", self.output_image_path);
        Ok(())
    }
}
