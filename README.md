# OCR_Neural_Network
![alt text](https://github.com/geekgirljoy/OCR_Neural_Network/blob/ca5dbcd3198dd36d2bbbbbc3731e450cac7ef7a4/Project%20Images/OCR.png)

OCR is a practical example of Optical Character Recognition using [FANN](https://github.com/bukka/php-fann). While this example is limited and does make mistakes, the concepts illustrated by OCR can be applied to a more robust stacked network that uses feature extraction and convolution layers to recognize text of any font in any size image.

**At the end of this series of tutorials you will be able to build Neural Networks using PHP that can read characters from images! I will be giving you actual working code!**

OCR is a practical example of Optical Character Recognition using [FANN](https://github.com/bukka/php-fann). While this example is limited and does make mistakes, the concepts illustrated by OCR can be applied to a more robust stacked network that uses feature extraction and convolution layers to recognize text of any font in any size image.

As mentioned this will be a series of posts so that I don’t overwhelm you guys with too much information all at once and so I don’t have to sit here and type [ad infinitum](https://en.wikipedia.org/wiki/Ad_infinitum) (infinite recursion).

😛

[OCR (Optical Character Recognition)](https://en.wikipedia.org/wiki/Optical_character_recognition) isn’t exactly a new subject but surprisingly its something that few computer scientists have actual experience building! Further, any examples you see are descriptions at best that frequently “devolve” into a math lesson that ultimately glosses over practical application and important implementation details!

Don’t get me wrong, I love math but that isn’t required to start learning. The FANN Library will act as an abstraction layer so we can focus on our data and objectives and not complex differential equations.

Additionally all the code will be thoroughly documented and intentionally simplified and referenced so that even a student with minimal experience can benefit. I happen to believe that Neural Networks are complex enough already, and the more people who know how to build and deploy these systems the faster we can find solutions to the most horrible problems we face as a species (currently incurable diseases, famine, wars over resources, the global energy crisis) need I go on?

So, I am going to provide you with the tools and basic knowledge of how to build POWERFUL artificial intelligence systems and deploy them to cloud servers, not to shabby eh? 😉

Not so you can go get rich building games and businesses (which you could easily do and that’s fine) but it is my very real hope that at lease some of you can apply these tools to help make the world a better place, start in your own community today!

In a very surreal way I am reminded of a lyric from the song [“I’d Love to Change the World” by Ten Years After](https://www.youtube.com/watch?v=ZyaFeDlJJAk)

*“I’d love to change the world, But I don’t know what to do, So I’ll leave it up to you-ooo-ooo”*

Its rather haunting in how true that actually is, isn’t it?

If you do happen to start a business using these tools and techniques or you simply appreciate the content that I create please support me on [Patreon](https://www.patreon.com/user?u=3969727) and please share this project with your followers, friends and coworkers on social media.

Now to begin you will need an environment to build your [ANN (Artificial Neural Network)](https://en.wikipedia.org/wiki/Artificial_neural_network), rather than reproduce the steps here you can follow this tutorial I wrote to get setup to work: [Getting started with Neural Networks using the FANN library, PHP and C9.io](https://geekgirljoy.wordpress.com/2016/07/12/getting-started-with-neural-networks-using-the-fann-library-php-and-c9-io/)

Go follow the setup process and come back, I’ll wait. 😉

So, in order for a Neural Network to be useful it needs a “problem space” (basically the thing we want it to learn or do) and a “training set” (examples of data similar to the data the ANN will encounter in the “problem space” but which already has a known value or solution). In this case, because we are teaching the ANN to read characters from images, we need a set of images with alpha-numeric characters in them.

Further, before getting started we have to make sure that we don’t violate the licensing of any training set we use so to keep things simple for this tutorial I will show you how to generate your own basic training data programmatically.

Because the training set will be generated programmatically rather than by hand we are excluding hand written characters from this example ANN however with the use of additional training sets and [convolution layers](https://en.wikipedia.org/wiki/Convolutional_neural_network) you could add that functionality to this neural network, however again, I want this example to be understandable by everyone.

So with that being said, lets begin by generating the training set of images!

We can use the [PHP GD library](http://php.net/manual/en/book.image.php)  which is a library used to manipulate images.

The GD Lib is almost always installed (compiled into php) by your host for you already and you probably have used it in the past (even if you didn’t realize it) for your other projects so I wont cover it in too much detail, suffice it to say its pretty east to get access to.

The images in our simple training set will be 10 pixels wide and 16 pixels high.

(10*16 = 160 pixels per image)

And in this example I will use only black and white for simplicity, white will be 1 and black will be 0. It’s completely arbitrary and you could reverse the colors if you wanted but white text on black seems to be high contrast so I decided to use that.

It’s also worth noting the importance of understanding that fundamentally there is no reason why you can’t use floating point values and represent the entire color spectrum or just gray scale.

Switching to a float would allow you to encode more information as a gradient however this will increase complexity of your ANN and you may require more hidden neurons, layers or both in addition to increased training epochs. In reality you would likely be MUCH better off adding [convolutions](https://en.wikipedia.org/wiki/Convolutional_neural_network) and building a more robust ANN.

This ANN does not use convolution layers and will make a few mistakes from time to time, my point is that this is a simplified, stripped down, easy to understand example however with a little work you could build this into a very robust OCR ANN.

So now lets look at a sample training image as well as dive into the code to generate them!
![alt text](https://github.com/geekgirljoy/OCR_Neural_Network/blob/ca5dbcd3198dd36d2bbbbbc3731e450cac7ef7a4/Project%20Images/TrainingImageInfoGraphic.png)

## generate_training_images.php

```php
<?php

/*
    NewImage( 
	          (int)$char, 
	          (mixed)$curr_image, 
			  (int)$x , 
			  (int)$y, 
			  (file resource)$logfile
			);
	
	Description: 
	
	This function actually creates the images. 
	
	It is written with an iterative process in mind where more than a single training image (batch creation) would be generated however a single image can be generated if you prefer.
	
	I have created the GenerateTrainingImages() function that will call this function for you so you *could*
	ignore this function but its the one that does the real work. 
		
	$char uses the chr() function to convert the number passed to an ASCII character.
	
	$curr_image is a mixed type variable used to name the image that is being generated. In this
	case (i.e how I used it in GenerateTrainingImages()) I have used it as a number so it's easy to 
	iterate over the image files using a simple for loop to programmatically create the images we
    need however you could create an array of strings or chars and use those instead however that is
	not quite as simple as the implementation shown here.
	
	$x & $y are direct pass-through variables for the x & y coordinate placement variables as defined
	in the imagestring() function documentation: http://php.net/manual/en/function.imagestring.php
    
	The main reason why I made them "pass-through" rather than hard coding them in the NewImage()
    function is simply that you may want to vary or stagger the placement of the character within the images.
	
	$logfile is a file resource variable that points to the file that will log results of generating
	the training images. You may want to use this file in later steps or for reference. Please note that 
	NewImage() does not open its own access to the file so you will need to open the file resource yourself
	prior to calling NewImage().
	
	
	Example NewImage() Usage:
	
	// Create $logfile resource
	$logfile = fopen("images/generate_images.log", "w") or die("Error: Unable to open: " . $logfile . '. Ending program.');
	
	// This will create a b/w image of an exclamation mark(!) named 0.png
	NewImage(33, 0, 1, 0, $logfile);
	
	// Close log file 
    fclose($logfile);
	
	
	References:	   
	   chr() - http://php.net/manual/en/function.chr.php
	   imagecreate() - http://us.php.net/manual/en/function.imagecreate.php
	   imagecolorallocate() - http://us.php.net/manual/en/function.imagecolorallocate.php
	   imagestring() - http://us.php.net/manual/en/function.imagestring.php
	   imagepng() - http://us.php.net/manual/en/function.imagepng.php
	   imagedestroy() - http://us.php.net/manual/en/function.imagedestroy.php
	   fwrite() - http://us.php.net/manual/en/function.fwrite.php
*/
function NewImage($char, $curr_image, $x , $y, $logfile){
    /* Size the images */
    $width = 10; // px
    $height= 16; // px
	
	/* Set the filename */
    $file_name = $curr_image . '.png';


    /* Create the image resource 
	
	   Note: The @ operator is used to suppress any errors generated by php expressions. 
	         http://us.php.net/manual/en/language.operators.errorcontrol.php
			 
	         I use it to suppress any errors from @imagecreate() and instead cast my own
			 error message by adding an "or die()" to the @imagecreate() statement.
	*/
    $image = @imagecreate($width, $height) or die("Error: Unable to Initialize Image Stream");
	
	
	/* Add colors to the image resource	
	   
	   Note: colors are defined by the RBG color model
	   https://en.wikipedia.org/wiki/RGB_color_model

	   RGB Black: (0, 0, 0)
	   RBG White: (255, 255, 255)	   
	*/
    $background_color = imagecolorallocate($image, 0, 0, 0);
    $text_color = imagecolorallocate($image, 255, 255, 255);
	
	/* Add $char to the image resource */
    imagestring($image, 5, $x, $y,  chr($char), $text_color);
	
    /* Draw the image buffer stream to the file */
    imagepng($image, './images/' . $file_name);
	
	/* Free the memory associated with the $image resource by using imagedestroy() */
    imagedestroy($image); 

    /* write to log file */ 
    fwrite($logfile, $curr_image . ' ' . chr($char). PHP_EOL);
	
	/* echo results and link to file for review */
	echo "<a href='images/$curr_image.png' target='_blank'>" . $curr_image . ".png</a> - " . chr($char) . " ...complete.<br>" . PHP_EOL;
}



/*
	GenerateTrainingImages(
	                        (NULL)
	                      );
	
	Description:
	
	This function manages the creation of the training images. Call this function to create a new training set. 

	
	Example GenerateTrainingImages() Usage:
	
	GenerateTrainingImages();
	
	
	References:
	file_exists() - http://us.php.net/manual/en/function.file-exists.php
	mkdir() - http://us.php.net/manual/en/function.mkdir.php
	fopen() - http://us.php.net/manual/en/function.fopen.php
	fclose() - http://us.php.net/manual/en/function.fclose.php
	
*/
function GenerateTrainingImages() { 

    /* Check if the "images" folder was already created */
    if (file_exists('images')) {
        echo "The images folder already exists, no changes to the images folder were made!<br>" . PHP_EOL;
    }
    else{ /* There is no "images" folder */
        echo "The images folder does not exists, creating one... ";
		
		/* try to create one with the correct folder permissions */
        if (!mkdir("images", 0755, true)) {
            die('fail! Ending program.'); /* Failed to create the folder */
        }else{
            echo 'success!<br>' . PHP_EOL; /* Successfully created the folder */
        }
    }

    /* Create log file resource to log the results of generating the training images */
    $logfile = fopen("images/generate_images.log", "w") or die("Unable to open: " . $logfile . '. Ending program.');

    /* Current number of generated images */    
    $curr_image = 0;

	
    /* 
        Training images set is defined in ASCII 
		https://en.wikipedia.org/wiki/ASCII
		
        start: ascii dec 33 (!) 
        stop:  ascii dec 126 (~) 
    */
	
	$start = 33;
	$stop = 126;
	$total = $stop - $start + 1; /* Add 1 because the count starts at 0 */
	echo "Starting batch creation of " .  $total . " images.<br><br>" . PHP_EOL;
	
    for($i = $start; $i <= $stop; $i++) {
        NewImage($i, $curr_image, 1, 0, $logfile);
        $curr_image++;
    }
	echo "<br>Batch complete.<br>" . PHP_EOL;
	echo "Log: <a href='images/generate_images.log' target='_blank'>generate_images.log</a><br>" . PHP_EOL;
    
	
    /* Close log file */
    fclose($logfile);
}

/* Kick the tires and light the fires! */
GenerateTrainingImages();

/* Announce completion and link to next step */
echo 'All Done! Now run <a href="generate_training_data.php">generate_training_data.php</a><br>' . PHP_EOL;

```


It looks like a lot of code, I know! If you are feeling overwhelmed delete my comments and you will see that the actual code is quite short and very simple.
If you run this code the following things will happen:

    1. A sub-folder called ‘images’ will be created for you where you ran generate_training_images.php.
    2. 94 training images will be generated for you and saved in the images sub-folder.
    3. A log file named ‘generate_images.log’ will be created.

At this point we are ready to use these images to create the training set that the Neural Network will use. We’ll cover that in the next post in this series.

If you would like to obtain a copy of this code from GitHub or fork this project to follow along as I release the code you can find this project here: [OCR on GitHub](https://github.com/geekgirljoy/OCR_Neural_Network)

If you have any questions, comments or trouble leave please leave it in the comments below and I will do my best to help out.

As always I hope you found this project both interesting and informative. Please Like, Comment & Share this post with your friends and followers on your social media platforms and don’t forget to click the follow button over on the top right of this page to get notified when I post something new.

Also please support me on [Patreon](https://www.patreon.com/user?u=3969727).

If would like to suggest a topic or project for an upcoming post feel free to [contact me](https://geekgirljoy.wordpress.com/contact/).

Much Love,
~Joy
