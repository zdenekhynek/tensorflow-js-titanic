/**
*	Normalize tensor to only include values from 0 to 1
*/
function normalizeTensor(tensor) {
	const max = tensor.max();
  const min = tensor.min();
  return tensor.sub(min).div(max.sub(min));
}

/**
*	Replace invalid numeric values
*/
function processQuantitativeParam(param) {
	return (!Number.isNaN(parseFloat(param)))? param : 0;
}


/**
*	Split shuffled data into a training and testing datasets
*/
function splitData(data) {
	tf.util.shuffle(data);

	const testCasesNum = 10;
	const train = data.slice(0, data.length - testCasesNum);
	const test = data.slice(data.length - testCasesNum);
	
	return [train, test];
}