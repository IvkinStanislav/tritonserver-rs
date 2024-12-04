use anyhow::Result;
use config_manager::{config, ConfigInit};
use image::{DynamicImage, GrayImage};
use tritonserver_rs::{Buffer, Request, Response};

use triton_examples::*;

#[tokio::main]
async fn main() -> Result<()> {
    let pipe = ImageTextPipe::parse()?;
    println!("{pipe:?}");

    run_pipeline(pipe).await
}

#[derive(Debug)]
#[config]
pub struct ImageTextPipe {
    #[source(clap(long, short), default = "DEFAULT_BACKEND_PATH.into()")]
    backend: String,
    #[source(clap(long, short), default = "\"models/text_detection\".into()")]
    model_path: String,
    #[source(clap(long, short), default = "\"data/input/traffic.jpg\".into()")]
    pub input_path: String,
    #[source(clap(long, short), default = "\"data/output\".into()")]
    pub output_path: String,
}

impl Pipeline for ImageTextPipe {
    fn backends_path(&self) -> String {
        self.backend.clone()
    }

    fn model_repo(&self) -> String {
        self.model_path.clone()
    }

    fn model_name(&self) -> String {
        "craft".into()
    }

    fn add_inputs(&mut self, request: &mut Request) -> Result<()> {
        let img = image::open(&self.input_path)?.resize_to_fill(
            1280,
            720,
            image::imageops::FilterType::Triangle,
        );

        let img_f32 = img
            .as_flat_samples_u8()
            .unwrap()
            .samples
            .iter()
            .map(|p| *p as f32)
            .collect::<Vec<_>>();
        let prepared_image = transpose(normalize_rgb(img_f32), 3);

        let img_buf = Buffer::from(prepared_image.as_slice());
        request.add_input_with_dims(
            "input",
            img_buf,
            [1, 3, img.width() as i64, img.height() as i64],
        )?;

        Ok(())
    }

    fn parse_result(&mut self, result: Response) -> Result<()> {
        let bboxes_and_links = result.get_output("output").unwrap();

        let width = bboxes_and_links.shape[1];
        let height = bboxes_and_links.shape[2];
        let bboxes_and_links = output_as_ref::<f32>(bboxes_and_links);

        // Divide by last dimention.
        let mut bboxes = Vec::new();
        let mut links = Vec::new();
        let mut box_turn = true;
        for value in bboxes_and_links {
            let pixel = if value > 0.2 { u8::MAX } else { u8::MIN };
            if box_turn {
                bboxes.push(pixel);
            } else {
                links.push(pixel);
            }

            box_turn = !box_turn;
        }

        let sum = bboxes.iter().map(|v| *v as f32).sum::<f32>() / 255.0;
        log::info!("BBoxes sum: {sum}");

        let img_bboxes =
            DynamicImage::ImageLuma8(GrayImage::from_raw(width as _, height as _, bboxes).unwrap());
        let img_links =
            DynamicImage::ImageLuma8(GrayImage::from_raw(width as _, height as _, links).unwrap());
        img_bboxes.save(self.output_path.clone() + "boxes.png")?;
        img_links.save(self.output_path.clone() + "links.png")?;

        Ok(())
    }
}

fn normalize_rgb(mut source: Vec<f32>) -> Vec<f32> {
    let mut channel_turn = 0;
    let colors_mean = [0.485, 0.456, 0.406];
    let color_variance = [0.229, 0.224, 0.225];

    for pixel in source.iter_mut() {
        *pixel -= colors_mean[channel_turn] * 255.0;
        *pixel /= color_variance[channel_turn] * 255.0;
        channel_turn = (channel_turn + 1) % 3;
    }

    source
}
