
/**
*	Translate two sexes into one-hot encoding
*/
function sexOneHotEncoding(sex) {
	return [
		sex === 'female' ? 1 : 0,
    sex === 'male' ? 1 : 0,
  ];
}

/**
*	Translate three classes into one-hot encoding
*/
function pclassOneHotEncoding(pclass) {
	return [
		pclass === 1 ? 1 : 0,
    pclass === 2 ? 1 : 0,
    pclass === 3 ? 1 : 0,
  ];
}

/**
*	Translate three classes into one-hot encoding
*/
function embarkedOneHotEncoding(embarked) {
	return [
		embarked === 'S' ? 1 : 0,
		embarked === 'C' ? 1 : 0,
		embarked === 'Q' ? 1 : 0,
  ];
}

/**
*	Extract features and labels from the dataset and convert them into a tensor
*/
function prepareData(data) {
	console.log('Preparing data');

	const xs = data.map((d) => [
		...sexOneHotEncoding(d.Sex),
		...pclassOneHotEncoding(d.Pclass),
		...embarkedOneHotEncoding(d.Embarked),
		processQuantitativeParam(d.Fare),
		processQuantitativeParam(d.Age),
		processQuantitativeParam(d.Parch),
		processQuantitativeParam(d.SibSp)
	]);
	const ys = data.map((d) => [d.Survived]);

	// Wrapping these calculations in a tidy will dispose any 
	// intermediate tensors.
	return tf.tidy(() => {
		let xsTensor = tf.tensor2d(xs, [xs.length, 12]);
		xsTensor = normalizeTensor(xsTensor);
		let ysTensor = tf.tensor2d(ys, [ys.length, 1]);
		ysTensor = normalizeTensor(ysTensor);

		return { xsTensor, ysTensor };
	});
}

/**
*	Define architecture of a neural network using the simplest 
* possible model configuration
*/
function createModel() {
	console.log('Creating model');

	//	Create a sequential model
	const model = tf.sequential();

	//	Add a single hidden layer, we're using 12 features
	//	to predict the species so input shape needs to be
	model.add(tf.layers.dense({ inputShape: [12], units: 8 }));

	//	Add an output layer, output is one hot encoding array with
	//	1 item, so need 1 units
	model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

	return model;
}	

/**
*	Train model is batches using training data with default paramaters
* for the model configuration (loss, metrics etc)
*/
async function trainModel(model, data) {
	console.log('Training model');
		
	const { xsTensor, ysTensor } = data;

	// Prepare the model for training
	model.compile({
		optimizer: tf.train.adam(),
		loss: tf.losses.meanSquaredError,
		metrics: ['mse'],
	});

	const batchSize = 300;
	const epochs = 100;

	return await model.fit(xsTensor, ysTensor, {
		batchSize,
		epochs,
		shuffle: true,
		callbacks: tfvis.show.fitCallbacks(
			{	name: 'Training Performance' },
			['loss', 'mse', 'accuracy'],
			{ height: 200, callbacks: ['onEpochEnd']
		})
	});
}

/**
*	Use trained model to make prediction on a testing data
* stored in a tensor.
*/
async function predict(model, testTensor) {
	console.log('Predicting');
	const pred = model.predict(testTensor);

	//	get typed array from prediction tensor and convert it
	//	to untyped
  const predArr = Array.from(pred.dataSync());
  pred.print();
  
  return predArr;
}

/**
*		Main function
*/
async function run() {
	console.log('Running script');
	//	Step 1. - split data into training and testing
	const [train, test] = splitData(titanicData);
	console.log(train, test);

	//	Step 2. - load and pre-process the data
	const trainData = prepareData(train);
	const testData = prepareData(test);

	//	Step 2. - create the model and train it
	const model = createModel();
	await trainModel(model, trainData);

	//	Step 3. - use the model for predictions
	const { xsTensor:testTensor } = testData;
	const predictions = await predict(model, testTensor);

	// Step 4. - compare predicitons with train data
	predictions.forEach((p, i) => {
		console.log(`For ${test[i].Survived}, prediction ${p}.`);
	})
}


document.addEventListener('DOMContentLoaded', run);
