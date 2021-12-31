const tf = require('@tensorflow/tfjs-node');

function normalized(data){ // x1 & x2 & x3
    x1 = (data[0] - 42.69172) / 10.62505
    x2 = (data[1] - 88.84096) / 19.00895
    x3 = (data[2] - 143.0686) / 22.97148
    return [x1, x2, x3]
}

function denormalized(data){
    y1 = (data[0] * 9.168212) + 74.72876
    y2 = (data[1] * 14.76342) + 49.87255
    y3 = (data[2] * 24.08864) + 159.8279
    return [y1, y2, y3]
}


async function predict(data){
    let in_dim = 3;
    
    data = normalized(data);
    shape = [1, in_dim];

    tf_data = tf.tensor2d(data, shape);

    try{
        // path load in public access => github
        const path = 'https://raw.githubusercontent.com/RiDraa/bot-41419010015/main/public/ex_model/model.json?token=5019360992:AAHp3nLbFCO5hle6XcqgMsjfFL0C5UnnWL4';
        const model = await tf.loadGraphModel(path);
        
        predict = model.predict(
                tf_data
        );
        result = predict.dataSync();
        return denormalized( result );
        
    }catch(e){
      console.log(e);
    }
}

module.exports = {
    predict: predict 
}
