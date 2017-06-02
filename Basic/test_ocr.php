<style>
	.blue{color:blue;}
	.green{color:green;}
	.red{color:red;}
</style>

<?php
/*
    OCR( 
	    (string) $img, 
	    (char) $expected, 
		(array) $input, 
		(array)$lookup_array, 
		(FANN neural network resource)$ann
	   );
	
	Description: 
	
	With this function I simply try to illustrate how you could test the OCR ANN. 
				
	$img is a string that should be the name to the training image you are reading from. It is
	used to output the image to the browser results.
	
	$expected is a char that you are actually testing for. eg  
	
	$input should be an array of inputs (encoded pixel data)
	
	$lookup_array should be an array of ASCII characters normalized as floating point values in increments of 0.01
	
	$ann should be a FANN neural network resource.
	
	References:   
	   global - http://php.net/language.variables.scope
	   PHP_EOL - http://php.net/manual/en/reserved.constants.php
	   fann_run() - http://php.net/manual/en/function.fann-run.php
	   floor() - http://php.net/manual/en/function.floor.php
	   count() - http://php.net/manual/en/function.count.php
*/
function OCR($img, $expected, $input, $lookup_array, $ann) {
	global $correct; // refer to the non local $correct variable
	$output = ""; 
	
	/* Display image for reference */
	$output .= "Image: <img src='images/$img'><br>" . PHP_EOL;

	// Run the ANN
	$calc_out = fann_run($ann, $input);
	
	$output .= 'Raw: ' .  $calc_out[0] . '<br>' . PHP_EOL;
	$output .= 'Trimmed: ' . floor($calc_out[0]*100)/100 . '<br>' . PHP_EOL;
	$output .= 'Decoded Symbol: ';
	
	/* What did the ANN think it saw? */
	for($i = 0; $i < count($lookup_array); $i++) {
       if( floor($lookup_array[$i][0]*100)/100 == floor($calc_out[0]*100)/100) {
	        $output .= $lookup_array[$i][1] . '<br>' . PHP_EOL;
	        $output .= "Expected: $expected <br>" . PHP_EOL;
	        $output .= 'Result: ';
	        if($expected == $lookup_array[$i][1]){
	        	$output .= '<span class="green">Correct!</span>';
				
				++$correct;
				
	        }else{
	        	$output .= '<span class="red">Incorrect!</span> <a href="train_ocr.php">Retrain OCR</a>';
	        }
		}
	}
	$output .= '<br><br>' . PHP_EOL;
	
	return $output;	
}


$total = 11; // How many images are to be tested
$correct = 0; // The count of how many images were correctly read by the ANN


/* Setup a resource that points to our ANN .net file */
$train_file = (dirname(__FILE__) . '/ocr_float.net');

/* Confirm the ANN exists */
if (!is_file($train_file))
	die('<span class="red">The file ocr_float.net has not been created!</span><a href="train_ocr.php">Train OCR</a>' . PHP_EOL);

/* Create the ANN resource */
$ocr_ann = fann_create_from_file($train_file);
if ($ocr_ann) {
	// Display the images we are testing (hard coded)
	?>
	<h1 class='blue'>OCR Test</h1>
	<strong>Testing: </strong>
	<img src='images/38.png'> <!-- G -->
	<img src='images/68.png'> <!-- e -->
	<img src='images/68.png'> <!-- e -->
	<img src='images/74.png'> <!-- k -->
	<img src='images/38.png'> <!-- G -->
	<img src='images/72.png'> <!-- i -->
	<img src='images/81.png'> <!-- r -->
	<img src='images/75.png'> <!-- l -->
	<img src='images/41.png'> <!-- J -->
	<img src='images/78.png'> <!-- o -->
	<img src='images/88.png'> <!-- y -->
	<br>
	<?php

	/* 
	    Create the lookup_array from ASCII
		https://en.wikipedia.org/wiki/ASCII
		
        start: ascii dec 33 (!) 
        stop:  ascii dec 126 (~) 
    */
	$result_lookup_array = array();
	$curr = 0.00;
	for($i = 33; $i <= 126; $i++) {
		array_push($result_lookup_array, array($curr, chr($i)));
		$curr+= 0.01;
	}
	

	// For simplicity sake I hardcoded these values below as there is no need to prove that we can read
	// the pixel data for each image (as we already did that in generate_training_data.php) however you 
	// can implement similar methodology to what as I did with GenerateTrainingData() to read pixel values
	// programmaticlly into an array rather than manually specifying it as I show here.
	
	$test_G = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	$test_e = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	$test_k = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	$test_i = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	$test_r = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	$test_l = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	$test_J = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	$test_o = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	$test_y = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	
       
	/* Test OCR and buffer results in the $output variable as string data */
	$output = "";
	$output .= OCR('38.png', 'G', $test_G, $result_lookup_array, $ocr_ann);
	$output .= OCR('68.png', 'e', $test_e, $result_lookup_array, $ocr_ann);
	$output .= OCR('68.png', 'e', $test_e, $result_lookup_array, $ocr_ann);
	$output .= OCR('74.png', 'k', $test_k, $result_lookup_array, $ocr_ann);
	$output .= OCR('38.png', 'G', $test_G, $result_lookup_array, $ocr_ann);
	$output .= OCR('72.png', 'i', $test_i, $result_lookup_array, $ocr_ann);
	$output .= OCR('81.png', 'r', $test_r, $result_lookup_array, $ocr_ann);
	$output .= OCR('75.png', 'l', $test_l, $result_lookup_array, $ocr_ann);
	$output .= OCR('41.png', 'J', $test_J, $result_lookup_array, $ocr_ann);
	$output .= OCR('78.png', 'o', $test_o, $result_lookup_array, $ocr_ann);
	$output .= OCR('88.png', 'y', $test_y, $result_lookup_array, $ocr_ann);
	
    // Determine how accurate the Neural Network is
    $percent_correct = round(($correct / $total) * 100, 2 );
    
	// Output the accuracy results
	echo "<strong>Results:</strong> $correct images correctly decoded out of $total. (<span class='"; 
	
	/* Add a css style to the percentage results */
	if($percent_correct < 70){echo "red'>";}
	elseif($percent_correct < 90){echo "blue'>";}
	else{echo "green'>";}
	
	/* Close css style span and offer link to retrain */
	echo $percent_correct . "%</span>)<br>Not good enough? <a href='train_ocr.php'>Retrain OCR</a><br><br>" ;
	
	/* display detailed results */
	echo "<h2 class='blue'>Details</h2>";
	echo $output;
	
	/* Free up memory associated with the OCR ANN resource. */ 
	fann_destroy($ocr_ann);
} else {
	die("<span class='red'>Invalid file format.</span>" . PHP_EOL);
}

?>
