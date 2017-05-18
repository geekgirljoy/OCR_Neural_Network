<?php

set_time_limit ( 300 ); // max run time 5 minutes (adjust as needed)

$num_input = 160;
$num_output = 1;
$num_layers = 3;
$num_neurons_hidden = 107; // 93
$desired_error = 0.00001;
$max_epochs = 5000000;
$epochs_between_reports = 10;

$ann = fann_create_standard($num_layers, $num_input, $num_neurons_hidden, $num_output);

if ($ann) {
	echo 'Training OCR... '; 
	fann_set_activation_function_hidden($ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output($ann, FANN_SIGMOID_SYMMETRIC);

	$filename = dirname(__FILE__) . "/ocr.data";
	if (fann_train_on_file($ann, $filename, $max_epochs, $epochs_between_reports, $desired_error))
		fann_save($ann, dirname(__FILE__) . "/ocr_float.net");

	fann_destroy($ann);
}

echo 'All Done! Now run <a href="test_ocr.php">Test OCR</a><br>' . PHP_EOL;
