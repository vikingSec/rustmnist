use std::{io::Read};
use rand::prelude::*;
const TRAIN_PATH : &str = "./data/mnist/train-images.idx3-ubyte";
const TRAIN_LBL_PATH : &str = "./data/mnist/train-labels.idx1-ubyte";
const INPUT_SZ : i32 = 784;
const HIDDEN_SZ : i32 = 256;
const OUTPUT_SZ : i32 = 10;
const LEARNING_RATE : f32 = 0.001;
const REPS : i32 = 20;
const BATCH_SZ : i32 = 64;
const IMG_SZ : i32 = 28;
const TRAIN_SPL : f32 = 0.8;
const RAND_MAX : f32 = 2147483647.0;

struct Layer {
    weights : Vec<f32>,
    biases : Vec<f32>,
    input_sz : i32,
    output_sz : i32,
}
impl Layer {

    fn new(insize : i32, outsize : i32) -> Layer {
        let scale = (2.0/insize as f32).sqrt();
        let n = insize * outsize;
        let mut weights = vec![0.0; n as usize];
        for i in 0..n as usize {
            weights[i] = (rand::random::<f32>() * RAND_MAX - 0.5) * scale * 2.;
        }

        Layer {
            input_sz : insize,
            output_sz : outsize,
            biases: vec![0.0; outsize as usize], 
            weights
        }
    }

}
struct Network {
    hidden: Layer,
    output : Layer
}

struct InputData<'a> {
   images: &'a str,
   labels: &'a str,
   num_imgs: i32
}
fn softmax(input : Vec<f32>) -> Vec<f32> {
    let mut output : Vec<f32> = vec![0.0; input.len() ];
    let mut sum = 0.0;
    let max = input.clone()
        .into_iter()
        .max_by(|a,b| a.total_cmp(b)).unwrap();
    let mut counter = 0;
    input.into_iter().map(|item| {
        let res = item.exp() - max;
        output[counter] = res;
        sum+=res;
        counter+=1;
    });
    counter = 0;
    while counter < output.len() {
        output[counter] = output[counter] / sum; 
        counter+=1;
    }

    output

}
fn read_labels() -> Vec<char> {
    let mut f = std::fs::File::open(TRAIN_LBL_PATH).expect("Training label path is bad");
    let mut buffer = [0u8; 8];
    f.read_exact(&mut buffer).expect("Failed to read header");

    let magic = i32::from_be_bytes(buffer[0..4].try_into().unwrap());
    let num_labels = i32::from_be_bytes(buffer[4..8].try_into().unwrap());
    println!("[{}] Number of labels: {}", magic, num_labels);

    let mut labels = Vec::with_capacity(num_labels as usize);
    for i in 0..num_labels {
        let mut label_byte = [0u8; 1];
        f.read_exact(&mut label_byte).expect("Failed to read label");
        let label = (label_byte[0] as u8 + b'0') as char;
        labels.push(label);
    }

    labels
}

fn read_images() -> Vec<char> {
    let mut f = std::fs::File::open(TRAIN_PATH).expect("Training image path is bad");
    let mut buffer = [0u8; 16];
    f.read_exact(&mut buffer).expect("Failure reading header bytes");
    let magic = i32::from_be_bytes(buffer[0..4].try_into().unwrap());
    let num_images = i32::from_be_bytes(buffer[4..8].try_into().unwrap());
    let num_rows = i32::from_be_bytes(buffer[8..12].try_into().unwrap());
    let num_cols = i32::from_be_bytes(buffer[12..16].try_into().unwrap());
    println!("[{}] Number of images: {}, Dimensions: {}x{}", magic, num_images, num_rows, num_cols);
    
    let mut ret: Vec<char> = Vec::with_capacity((num_images * num_rows * num_cols) as usize);
    for _ in 0..num_images {
        let mut image = vec![0u8; (num_rows * num_cols) as usize];
        f.read_exact(&mut image).expect("Failed to read a new image");
        for pixel in image {
            ret.push((pixel as char).to_ascii_lowercase());
        }
    }
    ret
}





fn main() {
    let start_time = std::time::Instant::now();

    let labels = read_labels();
    let images = read_images();
    let mut net = Network {
        hidden : Layer::new(INPUT_SZ, HIDDEN_SZ),
        output : Layer::new(HIDDEN_SZ, OUTPUT_SZ)
    };

    let duration = start_time.elapsed();
    println!("Setup time: {:?}", duration);
}
