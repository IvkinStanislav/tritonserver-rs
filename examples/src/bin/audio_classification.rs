use std::{collections::HashMap, path::PathBuf};

use anyhow::Result;
use config_manager::{config, ConfigInit};
use hound::{SampleFormat, WavReader, WavSpec};
use tritonserver_rs::{Buffer, Request, Response};

use triton_examples::*;

#[tokio::main]
async fn main() -> Result<()> {
    let pipe = AudioPipe::parse()?;
    println!("{pipe:?}");

    run_pipeline(pipe).await
}

#[derive(Debug)]
#[config]
struct AudioPipe {
    #[source(clap(long, short), default = "DEFAULT_BACKEND_PATH.into()")]
    backend: String,
    #[source(clap(long, short), default = "\"models/audio_classification\".into()")]
    model_path: String,
    #[source(
        clap(long, short),
        default = "\"data/input/fight-club-trailer.wav\".into()"
    )]
    input_wav: String,
    #[source(clap(long, short), default = "\"data/input/audio_classes.csv\".into()")]
    classes_csv_path: String,
}

impl Pipeline for AudioPipe {
    fn backends_path(&self) -> String {
        self.backend.clone()
    }

    fn add_inputs(&mut self, request: &mut Request) -> Result<()> {
        let wav = hound::WavReader::open(&self.input_wav)?;

        let samples = prepare_audio(wav)?;

        let content = Buffer::from(samples.as_slice());
        request.add_input_with_dims("waveform", content, [samples.len() as _])?;
        Ok(())
    }

    fn model_name(&self) -> String {
        "yamnet".into()
    }

    fn model_repo(&self) -> String {
        self.model_path.clone()
    }

    fn parse_result(&mut self, result: Response) -> Result<()> {
        let scores_names = parse_csv(PathBuf::from(&self.classes_csv_path))?;

        let scores = result.get_output("output_0").unwrap();
        let scores = output_as_ref::<f32>(scores);
        let frames_scores = scores.chunks(521);
        let frames_max_scores = frames_scores
            .map(|c| arg_max_n(c, 2))
            .filter_map(|classes_ids| {
                Some((
                    scores_names.get(&classes_ids[0])?,
                    scores_names.get(&classes_ids[1])?,
                ))
            })
            .collect::<Vec<_>>();
        log::info!("Max scores for each frame: {frames_max_scores:?}");

        Ok(())
    }
}

fn prepare_audio<R: std::io::Read>(mut wav: WavReader<R>) -> Result<Vec<f32>> {
    const TARGET_SAMPLE_RATE: u32 = 16_000;

    let WavSpec {
        channels,
        sample_rate,
        bits_per_sample: _,
        sample_format,
    } = wav.spec();

    let samples: Result<Vec<f32>, _> = if sample_format == SampleFormat::Float {
        wav.samples::<f32>().collect()
    } else {
        wav.samples::<i16>()
            .map(|sample| sample.map(|s| s as f32 / i16::MAX as f32))
            .collect()
    };

    let samples = if channels == 2 {
        samples?
            .chunks_exact(2)
            .map(|left_right| (left_right[0] + left_right[1]) / 2.0)
            .collect()
    } else {
        samples?
    };

    let rate_ratio = sample_rate as f32 / TARGET_SAMPLE_RATE as f32;
    let samples = if rate_ratio > 1.0 {
        let mut keep_frame_counter = 1.0;
        samples
            .into_iter()
            .enumerate()
            .filter_map(|(i, sample)| {
                if (keep_frame_counter * rate_ratio) as usize == i {
                    keep_frame_counter += 1.0;
                    Some(sample)
                } else {
                    None
                }
            })
            .collect()
    } else if rate_ratio == 1.0 {
        samples
    } else {
        let rate_ratio = 1.0 / rate_ratio;
        let mut copy_frame_counter = rate_ratio;
        samples.into_iter().fold(Vec::new(), |mut acc, new| {
            while copy_frame_counter >= 1.0 {
                acc.push(new);
                copy_frame_counter -= 1.0;
            }
            copy_frame_counter += rate_ratio;
            acc
        })
    };

    Ok(samples)
}

fn arg_max_n<T: PartialOrd + Copy>(slice: &[T], n: usize) -> Vec<usize> {
    let mut sorted: Vec<_> = slice.iter().enumerate().collect();
    sorted.sort_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap().reverse());

    sorted.into_iter().map(|(pos, _)| pos).take(n).collect()
}

#[derive(Debug, serde::Deserialize)]
struct AudioCsvValue {
    index: usize,
    #[allow(dead_code)]
    mid: String,
    display_name: String,
}

fn parse_csv(path: PathBuf) -> Result<HashMap<usize, String>> {
    let mut reader = csv::Reader::from_path(path)?;

    Ok(reader
        .deserialize::<AudioCsvValue>()
        .filter_map(|result| {
            result
                .ok()
                .map(|record| (record.index, record.display_name))
        })
        .collect())
}
