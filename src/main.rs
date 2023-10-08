use clap::Parser;
use eyre::Result;
use indicatif::ProgressBar;
use llm::{Model, ModelArchitecture};
use log::{info, Level, LevelFilter};
use owo_colors::{AnsiColors, OwoColorize};
use smol::{
    future::yield_now,
    io::{self, BufReader},
    net::{TcpListener, TcpStream},
    prelude::*,
    LocalExecutor,
};
use std::{
    cell::OnceCell, collections::HashMap, convert::Infallible, fs, io::Write, net::SocketAddr,
    path::PathBuf, time::Instant,
};
use string_template::Template;

#[cfg(all(feature = "cublas", feature = "clblast"))]
compile_error!("only at most one of either cublas or clblast can be enabled at once!");

fn main() -> Result<()> {
    color_eyre::install()?;
    pretty_env_logger::formatted_builder()
        .filter_level(LevelFilter::Info)
        .parse_default_env()
        .format(|fmt, rec| {
            writeln!(
                fmt,
                "{} {}",
                rec.target().bold().color(match rec.level() {
                    Level::Trace => AnsiColors::White,
                    Level::Debug => AnsiColors::Cyan,
                    Level::Info => AnsiColors::Green,
                    Level::Warn => AnsiColors::Yellow,
                    Level::Error => AnsiColors::Red,
                }),
                rec.args()
            )
        })
        .init();

    let args = Args::parse();

    let bar = OnceCell::new();
    let tensor_load_msg = OnceCell::new();
    let mut last_tensor_loaded = 0;
    info!("loading model");
    let model = llm::load_dynamic(
        args.model_arch,
        &args.model,
        llm::TokenizerSource::Embedded,
        llm::ModelParameters {
            #[cfg(any(feature = "cublas", feature = "clblast"))]
            use_gpu: true,
            #[cfg(not(any(feature = "cublas", feature = "clblast")))]
            use_gpu: false,

            #[cfg(any(feature = "cublas", feature = "clblast"))]
            gpu_layers: args.gpu_layers,
            #[cfg(not(any(feature = "cublas", feature = "clblast")))]
            gpu_layers: None,

            ..Default::default()
        },
        |progress| match progress {
            llm::LoadProgress::HyperparametersLoaded => info!("loaded hyperparams"),
            llm::LoadProgress::ContextSize { bytes } => info!("context size: {bytes}B"),
            llm::LoadProgress::TensorLoaded {
                current_tensor,
                tensor_count,
            } => {
                tensor_load_msg.get_or_init(|| info!("loading tensors"));
                bar.get_or_init(|| ProgressBar::new(tensor_count as u64))
                    .inc(current_tensor as u64 - last_tensor_loaded);
                last_tensor_loaded = current_tensor as u64;
            }
            llm::LoadProgress::LoraApplied { name, source } => {
                info!("applied LoRA {name} from {source:?}");
            }
            llm::LoadProgress::Loaded {
                file_size,
                tensor_count,
            } => {
                bar.get()
                    .expect("progress bar should have been initialized by now")
                    .finish_and_clear();
                info!("loaded model ({file_size}B, {tensor_count} tensors)");
            }
        },
    )?;
    let prompt_fmt = Template::new(&fs::read_to_string(args.prompt_template)?);

    let executor = LocalExecutor::new();
    smol::block_on(executor.run(async {
        let socket = TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], args.port))).await?;

        while let Some(Ok(stream)) = socket.incoming().next().await {
            executor
                .spawn(handler(
                    &*model,
                    stream,
                    args.batch_size,
                    args.threads,
                    &prompt_fmt,
                ))
                .detach();
        }

        Ok(())
    }))
}

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    model: PathBuf,
    #[arg(short = 'a', long)]
    model_arch: Option<ModelArchitecture>,
    #[arg(short = 'T', long)]
    prompt_template: PathBuf,
    #[arg(short, long, default_value_t = 3000)]
    port: u16,
    #[cfg(any(feature = "cublas", feature = "clblast"))]
    #[arg(short, long)]
    gpu_layers: Option<usize>,
    #[arg(short, long, default_value_t = 8)]
    batch_size: usize,
    #[arg(short, long, default_value_t = 2)]
    threads: usize,
}

async fn handler(
    model: &dyn Model,
    stream: TcpStream,
    n_batch: usize,
    n_threads: usize,
    prompt_fmt: &Template,
) -> Result<()> {
    let mut session = model.start_session(llm::InferenceSessionConfig {
        n_batch,
        n_threads,
        ..Default::default()
    });

    let (reader, mut writer) = io::split(stream);
    let mut lines = BufReader::new(reader).lines();

    loop {
        let Some(Ok(next_line)) = lines.next().await else {
            return Ok(());
        };

        let mut args = HashMap::new();
        args.insert("prompt", next_line.as_str());
        let prompt = prompt_fmt.render(&args);

        let timer = Instant::now();

        for word in prompt.split_whitespace() {
            session.feed_prompt(model, word, &mut llm::OutputRequest::default(), |_| {
                Ok::<_, Infallible>(llm::InferenceFeedback::Continue)
            })?;
            yield_now().await;
        }

        loop {
            let Ok(tok) = session.infer_next_token(
                model,
                &llm::InferenceParameters::default(),
                &mut llm::OutputRequest::default(),
                &mut rand::thread_rng(),
            ) else {
                break;
            };

            writer.write_all(&tok).await?;
            yield_now().await;
        }

        let elapsed = timer.elapsed();

        writer
            .write_all(format!("\r\n(took {:.2}s)\r\n", elapsed.as_secs_f64()).as_bytes())
            .await?;
    }
}
