use std::{collections::HashMap, fs::File, io::Read, path::PathBuf};

use ab_glyph::FontVec;
use anyhow::{anyhow, Result};
use config_manager::{config, ConfigInit};
use image::{DynamicImage, Rgba};
use imageproc::rect::Rect;
use tritonserver_rs::{Buffer, Request, Response};

use triton_examples::*;

const TARGET_WIDTH: usize = 640;
const TARGET_HEIGHT: usize = 640;

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::parse()?;
    println!("{config:?}");

    let pipe = ImagePipe::new(config)?;

    run_pipeline(pipe).await
}

#[derive(Debug)]
#[config]
struct Config {
    #[source(clap(long, short), default = "DEFAULT_BACKEND_PATH.into()")]
    backend: String,
    #[source(clap(long, short), default = "\"models/image_detection\".into()")]
    model_path: String,
    #[source(clap(long, short), default = "\"data/input/traffic.jpg\".into()")]
    image_path: String,
    #[source(clap(long, short), default = "Some(\"data/input/ARIAL.TTF\".into())")]
    fonts_path: Option<String>,
    #[source(
        clap(long, short),
        default = "Some(\"data/input/image_classes.csv\".into())"
    )]
    classes_csv_path: Option<String>,
    #[source(clap(long, short), default = "\"data/output/traffic_bb.jpg\".into()")]
    output_path: String,
}

#[derive(Debug)]
pub struct ImagePipe {
    cfg: Config,

    img: DynamicImage,
    w_ratio: f32,
    h_ratio: f32,
    font: Option<FontVec>,
    classes: HashMap<usize, String>,
}

impl ImagePipe {
    fn new(cfg: Config) -> Result<Self> {
        let img = image::open(&cfg.image_path)?;

        let w_ratio = img.width() as f32 / TARGET_WIDTH as f32;
        let h_ratio = img.height() as f32 / TARGET_HEIGHT as f32;
        let mut classes = HashMap::new();

        if let Some(csv_path) = &cfg.classes_csv_path {
            classes = parse_csv(csv_path.into())?;
        }

        let font = if let Some(font_path) = &cfg.fonts_path {
            let bytes: Result<Vec<u8>, _> = File::open(font_path)?.bytes().collect();
            Some(FontVec::try_from_vec(bytes?)?)
        } else {
            None
        };

        Ok(Self {
            cfg,
            img,
            w_ratio,
            h_ratio,
            font,
            classes,
        })
    }

    fn draw_bbox(&mut self, object: Detection) -> Result<()> {
        let left = (object.bbox[0] * self.w_ratio) as i32;
        let top = (object.bbox[1] * self.h_ratio) as i32;
        let width = (object.bbox[2] * self.w_ratio).max(1.0) as u32;
        let height = (object.bbox[3] * self.h_ratio).max(1.0) as u32;

        let class_color = Rgba(
            random_color::RandomColor::new()
                .seed(object.class)
                .to_rgba_array(),
        );

        imageproc::drawing::draw_hollow_rect_mut(
            &mut self.img,
            Rect::at(left, top).of_size(width, height),
            class_color,
        );

        if let Some(font) = &self.font {
            let class = self
                .classes
                .get(&(object.class as _))
                .ok_or(anyhow!("Unknown class id : {}", object.class))?;
            imageproc::drawing::draw_text_mut(
                &mut self.img,
                class_color,
                left - (10.0 * self.h_ratio) as i32,
                top - (10.0 * self.w_ratio) as i32,
                10.0 * (self.h_ratio * self.w_ratio).sqrt(),
                font,
                &format!("{class}: conf {:.2}", object.score),
            );
        }

        Ok(())
    }
}

impl Pipeline for ImagePipe {
    fn backends_path(&self) -> String {
        self.cfg.backend.clone()
    }

    fn model_repo(&self) -> String {
        self.cfg.model_path.clone()
    }

    fn model_name(&self) -> String {
        "yolov8_ensemble".into()
    }

    fn add_inputs(&mut self, request: &mut Request) -> Result<()> {
        let img: Vec<f32> = self
            .img
            .resize_exact(
                TARGET_WIDTH as _,
                TARGET_HEIGHT as _,
                image::imageops::FilterType::Triangle,
            )
            .as_flat_samples_u8()
            .unwrap()
            .samples
            .iter()
            .map(|s| *s as f32 / 255.0)
            .collect();

        let img_buf = Buffer::from(transpose(img, 3).as_slice());
        request.add_input("images", img_buf)?;

        Ok(())
    }

    fn parse_result(&mut self, result: Response) -> Result<()> {
        let num_detects = result.get_output("num_detections").unwrap();
        let bboxes = result.get_output("detection_boxes").unwrap();
        let scores = result.get_output("detection_scores").unwrap();
        let classes = result.get_output("detection_classes").unwrap();

        let num_detects = output_as_ref::<i32>(num_detects)[0] as usize;
        if num_detects == 0 {
            log::info!("Empty frame");
        } else {
            log::info!("Got {num_detects} detections");

            let bboxes = output_as_ref::<f32>(bboxes);
            let classes = output_as_ref::<i32>(classes);
            let scores = output_as_ref::<f32>(scores);
            for i in 0..num_detects {
                let object = Detection {
                    bbox: bboxes[i * 4..(i + 1) * 4].try_into()?,
                    class: classes[i],
                    score: scores[i],
                };
                self.draw_bbox(object)?;
            }

            self.img.save(&self.cfg.output_path)?;
        }

        Ok(())
    }
}

struct Detection {
    bbox: [f32; 4],
    class: i32,
    score: f32,
}

#[derive(Debug, serde::Deserialize)]
struct ImageCsvValue {
    class_id: usize,
    class_name: String,
}

fn parse_csv(path: PathBuf) -> Result<HashMap<usize, String>> {
    let mut reader = csv::Reader::from_path(path)?;

    Ok(reader
        .deserialize::<ImageCsvValue>()
        .filter_map(|result| {
            result
                .ok()
                .map(|record| (record.class_id, record.class_name))
        })
        .collect())
}
