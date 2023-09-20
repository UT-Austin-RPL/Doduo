---
layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>
<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>Doduo</title>


<!-- <meta property="og:image" content="images/teaser_fb.jpg"> -->
<meta property="og:title" content="TITLE">

<script src="./src/popup.js" type="text/javascript"></script>

<!-- Global site tag (gtag.js) - Google Analytics -->

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: "Titillium Web","HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;
    font-weight:300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
  }
  
  h1 {
    font-weight:300;
  }
  h2 {
    font-weight:300;
  }
  
IMG {
  PADDING-RIGHT: 0px;
  PADDING-LEFT: 0px;
  <!-- FLOAT: justify; -->
  PADDING-BOTTOM: 0px;
  PADDING-TOP: 0px;
   display:block;
   margin:auto;  
}
#primarycontent {
  MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
1000px }
BODY {
  TEXT-ALIGN: center
}
hr
  {
    border: 0;
    height: 1px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }

  pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
}
table 
	{
	width:800
	}

#container {
      display: flex;
      margin: 10px;
    }

    .image-container {
      flex: 1;
      margin: 10px;
      position: relative;
    }

    .dot {
      position: absolute;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      border: 2px solid white;
    }

    #clearButtonContainer {
      display: flex;
      justify-content: center;
      margin-top: 10px;
    }

    #clearButton {
      padding: 10px 20px;
      font-family: "Titillium Web", "HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;
      font-weight: 300;
      font-size: 18px;
      border-radius: 20px;
      border: none;
      background-color: #f2f2f2;
      transition: background-color 0.3s ease;
    }

    #clearButton:hover {
      background-color: #e0e0e0;
    }
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


<!-- <link rel="apple-touch-icon" sizes="120x120" href="/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">
<link rel="manifest" href="/site.webmanifest">
<link rel="mask-icon" href="/safari-pinned-tab.svg" color="#5bbad5">
<meta name="msapplication-TileColor" content="#da532c">
<meta name="theme-color" content="#ffffff"> -->

<link rel="shortcut icon" type="image/x-icon" href="favicon.ico">
</head>

<body data-gr-c-s-loaded="true">

<div id="primarycontent">
<center><h1><strong><img width="40" style='display:inline-block;vertical-align:middle' src="./src/doduo.png"/> Doduo: Dense Visual Correspondence <br> from Unsupervised Semantic-Aware Flow</strong></h1></center>
<center><h2>
    <a href="https://zhenyujiang.me/">Zhenyu Jiang</a>&nbsp;&nbsp;&nbsp;
    <a href="https://hwjiang1510.github.io/">Hanwen Jiang</a>&nbsp;&nbsp;&nbsp; 
    <a href="https://cs.utexas.edu/~yukez">Yuke Zhu</a>&nbsp;&nbsp;&nbsp;
   </h2>
    <center><h2>
        <a href="https://www.cs.utexas.edu/">The University of Texas at Austin</a>&nbsp;&nbsp;&nbsp; 		
    </h2></center>
<!-- <center><h2>
        CVPR 2022 Oral Presentation&nbsp;&nbsp;&nbsp; 		
    </h2></center> -->
	<center><h2><a href="src/Doduo_ICRA_2024.pdf">Paper</a> | <a href="https://github.com/UT-Austin-RPL/Doduo">Code</a> | <a href="https://huggingface.co/stevetod/doduo/">Huggingface Model</a> </h2></center>


<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
Dense visual correspondence plays a vital role in robotic perception. This work focuses on establishing the dense correspondence between a pair of images that captures dynamic scenes undergoing substantial transformations. We introduce Doduo to learn general dense visual correspondence from in-the-wild images and videos without ground truth supervision. Given a pair of images, it estimates the dense flow field encoding the displacement of each pixel in one image to its corresponding pixel in the other image. Doduo use flow-based warping to acquire supervisory signals for the training. Incorporating semantic priors with self-supervised flow training, Doduo produces accurate dense correspondence robust to the dynamic changes of the scenes. 
Trained on an in-the-wild video dataset, Doduo illustrates superior performance on point-level correspondence estimation over existing self-supervised correspondence learning baselines. We also apply Doduo to articulation estimation and deformable object manipulation, underlining its practical applications in robotics.
</p></td></tr></table>
</p>
</div>
</p>

<br><hr> <h1 align="center">Dense Visual Correspondence</h1> <!-- <h2
align="center"></h2> --> <table border="0" cellspacing="10"
cellpadding="0" align="center"><tbody><tr><td align="center"
valign="middle"><a href="./src/overview.png"> <img
src="./src/overview.png" style="width:100%;"> </a></td>
</tr> </tbody> </table>

<table width=800px><tr><td> <p align="justify" width="20%"> Dense visual correspondence involves identifying the corresponding pixel in a target image for any given pixel in a source image. Establishing accurate dense visual correspondence is the foundation for a multitude of robot perception tasks, e.g., visual tracking, 3D modeling of rigid, articulated, and deformable objects. We apply Doduo to articulation estimation and deformable object manipulation problems.</p></td></tr></table>

  
<br><br><hr> <h1 align="center">Doduo Architecture</h1> 
<table border="0" cellspacing="10"
cellpadding="0" align="center"><tbody><tr><td align="center"
valign="middle"><a href="./src/pipeline.png"> <img
src="./src/pipeline.png" style="width:70%;"> </a></td>
</tr> </tbody> </table>

<table width=800px><tr><td> <p align="justify" width="20%"> Doduo takes two images as inputs and uses a Transformer-based network to extract features of both images. Then we construct a 4D cost volume of all pairs of feature pixels and predict a dense semantic-aware flow field given the cost volume. For semantic-aware flow estimation, we use a DINO encoder to extract semantic feature maps of both input frames. According to the similarity of the semantic feature map, we compute a matching candidate mask for each of the feature points of the source frame, and integrate this information during flow estimation using masked Softmax. </p></td></tr></table>
<br>

<hr>


<h1 align="center"> Point Correspondence </h1>

<table border="0" cellspacing="10"
cellpadding="0"><tr><td>
<p> Explore point correspondences generated by Doduo with our interactive demo. Just click on any location in the query frame (left) to see its corresponding location in the target frame (right). Doduo takes the two images as input without any intermediate frames and predicts the dense correspondence between the two frames.</p></td></tr></table>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr><td>

<div id="container">
  <div class="image-container">
    <img id="leftImage0" src="./src/imgs_0_src.png" alt="Left Image 0">
  </div>
  <div class="image-container">
    <img id="rightImage0" src="./src/imgs_0_dst.png" alt="Right Image 0">
  </div>
</div>

<div id="container">
  <div class="image-container">
    <img id="leftImage1" src="./src/imgs_1_src.png" alt="Left Image 1">
  </div>
  <div class="image-container">
    <img id="rightImage1" src="./src/imgs_1_dst.png" alt="Right Image 1">
  </div>
</div>

<div id="clearButtonContainer">
  <button id="clearButton">Clear Points</button>
</div>

<script>
  // Generate a random color
  function getRandomColor() {
    var letters = "0123456789ABCDEF";
    var color = "#";
    for (var i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
  }

  // Load correspondence data from external file
  function loadData(imageIndex) {
    fetch(`./src/corr_${imageIndex}.json`)
      .then(response => response.json())
      .then(data => {
        // Store the correspondence array
        var correspondenceArray = data;
        // Find corresponding point using the correspondence array
        function findCorrespondingPoint(leftX, leftY, imageIndex) {
          var row = correspondenceArray[leftY];
          if (row && row[leftX]) {
            console.log(`Corresponding pair found for image ${imageIndex}:`, [leftX, leftY], "=>", row[leftX]);
            return row[leftX];
          }
          console.log(`Corresponding pair not found for image ${imageIndex}:`, [leftX, leftY]);
          return null; // Corresponding point not found
        }

        // Left image click event handler
        document.getElementById(`leftImage${imageIndex}`).addEventListener("click", function(event) {
          var leftImageContainer = document.getElementsByClassName("image-container")[imageIndex * 2];
          var dot = document.createElement("div");
          dot.className = "dot";
          dot.style.backgroundColor = getRandomColor();

          var rect = leftImageContainer.getBoundingClientRect();

          var leftX = Math.floor(event.clientX - rect.left);
          var leftY = Math.floor(event.clientY - rect.top);

          dot.style.left = leftX + "px";
          dot.style.top = leftY + "px";

          leftImageContainer.appendChild(dot);

          // Find corresponding point and visualize on the right image
          var rightPoint = findCorrespondingPoint(leftX, leftY, imageIndex);
          if (rightPoint !== null) {
            var rightX = rightPoint[0];
            var rightY = rightPoint[1];

            var rightImageContainer = document.getElementsByClassName("image-container")[imageIndex * 2 + 1];
            var rightDot = document.createElement("div");
            rightDot.className = "dot";
            rightDot.style.backgroundColor = dot.style.backgroundColor;
            rightDot.style.left = rightX + "px";
            rightDot.style.top = rightY + "px";

            rightImageContainer.appendChild(rightDot);
          }
        });
      })
      .catch(error => {
        console.error(`Failed to load correspondence data for image ${imageIndex}:`, error);
      });
  }

  // Function to clear points
  function clearPoints() {
    var dots = document.getElementsByClassName("dot");
    while (dots.length > 0) {
      dots[0].remove();
    }
  }

  // Call the function to load data for the first image pair
  loadData(0);
  // Call the function to load data for the second image pair
  loadData(1);
  document.getElementById("clearButton").addEventListener("click", clearPoints);
</script>


  <!-- <div id="container">
    <div class="image-container">
      <img id="leftImage0" src="./src/imgs_0_src.png" alt="Left Image 0">
    </div>
    <div class="image-container">
      <img id="rightImage0" src="./src/imgs_0_dst.png" alt="Right Image 0">
    </div>
  </div>

  <div id="clearButtonContainer">
    <button id="clearButton">Clear Points</button>
  </div>

  <script>
    // Load correspondence data from external file
    fetch("./src/corr_0.json")
      .then(response => response.json())
      .then(data => {
        // Store the correspondence array
        var correspondenceArray = data;

        // Left image click event handler
        document.getElementById("leftImage0").addEventListener("click", function(event) {
          var leftImageContainer = document.getElementsByClassName("image-container")[0];
          var dot = document.createElement("div");
          dot.className = "dot";
          dot.style.backgroundColor = getRandomColor();

          var rect = leftImageContainer.getBoundingClientRect();

          var leftX = Math.floor(event.clientX - rect.left);
          var leftY = Math.floor(event.clientY - rect.top);


          dot.style.left = leftX + "px";
          dot.style.top = leftY + "px";

          leftImageContainer.appendChild(dot);

          // Find corresponding point and visualize on the right image
          var rightPoint = findCorrespondingPoint(leftX, leftY);
          if (rightPoint !== null) {
            var rightX = rightPoint[0];
            var rightY = rightPoint[1];

            var rightImageContainer = document.getElementsByClassName("image-container")[1];
            var rightDot = document.createElement("div");
            rightDot.className = "dot";
            rightDot.style.backgroundColor = dot.style.backgroundColor;
            rightDot.style.left = rightX + "px";
            rightDot.style.top = rightY + "px";

            rightImageContainer.appendChild(rightDot);
          }
        });

        // Clear points button click event handler
        document.getElementById("clearButton").addEventListener("click", function() {
          var imageContainers = document.getElementsByClassName("image-container");
          var dots = document.getElementsByClassName("dot");

          while (dots.length > 0) {
            dots[0].remove();
          }
        });

        // Generate a random color
        function getRandomColor() {
          var letters = "0123456789ABCDEF";
          var color = "#";
          for (var i = 0; i < 6; i++) {
            color += letters[Math.floor(Math.random() * 16)];
          }
          return color;
        }

        // Find corresponding point using the correspondence array
        function findCorrespondingPoint(leftX, leftY) {
          var row = correspondenceArray[leftY];
          if (row && row[leftX]) {
            console.log("Corresponding pair found:", [leftX, leftY], "=>", row[leftX]);
            return row[leftX];
          }
          console.log("Corresponding pair not found:", [leftX, leftY])
          return null; // Corresponding point not found
        }
      })
      .catch(error => {
        console.error("Failed to load correspondence data:", error);
      });
  </script> -->

</td></tr>
</tbody>
</table>


<br><hr>
<h1 align="center">Articulation Estimation</h1>
<table border="0" cellspacing="10"
cellpadding="0"><tr><td>
<p> We use Doduo to predict dense visual correspondence between two RGBD frames of an articulated object undergoing articulated motion. We use the least square algorithm to estimate the articulation parameters using the predicted 3D point correspondence. </p></td></tr></table>

<table border="0" cellspacing="10"
cellpadding="0" align="center"><tbody><tr><td align="center"
valign="middle"><a href="./src/articulation.png"> <img
src="./src/articulation.png" style="width:100%;"> </a></td>
</tr> </tbody> </table>


<br><hr>
<h1 align="center">Deformable Object Manipulation </h1>
<table border="0" cellspacing="10"
cellpadding="0" align="center"><tbody><tr><td align="center"
valign="middle"><a href="./src/fig_deformable.png"> <img
src="./src/fig_deformable.png" style="width:100%;"> </a></td>
</tr> </tbody> </table>

<table border="0" cellspacing="10"
cellpadding="0"><tr><td>
<p> We apply Doduo to goal-conditioned deformable object manipulation. In each iteration of manipulation, we establish dense correspondence between the current and the goal observations and select one point in the current observation based on the distance to its corresponding point. Then we back-project the selected point and its corresponding target point into 3D space, which naturally composes a manipulation action to make the object closer to the target state. Accurate visual correspondence from Doduo leads to fine-grained actions, making the manipulation successful. </p></td></tr></table>
  

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
    <tr>
      <td align="center" valign="middle">
        <video muted autoplay loop controls width="100%">
          <source src="./video/deformable_video_sloth.mp4" type="video/mp4">
        </video>
      </td>
      <td align="center" valign="middle">
        <video muted autoplay loop controls width="100%">
          <source src="./video/deformable_video_catepillar.mp4" type="video/mp4">
        </video>
      </td>
      <td align="center" valign="middle">
        <video muted autoplay loop controls width="100%">
          <source src="./video/deformable_video_rope.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
  </tbody>
</table>


<br>



<hr>
<!-- <table align=center width=800px> <tr> <td> <left> -->
<center><h1>Citation</h1></center>

<table align=center width=800px>
              <tr>
                  <td>
                  <left>
<pre><code style="display:block; overflow-x: auto">@inproceedings{jiang2023doduo,
   title={Doduo: Dense Visual Correspondence from Unsupervised Semantic-Aware Flow},
   author={Jiang, Zhenyu and Jiang, Hanwen and Zhu, Yuke},
   booktitle={TODO},
   year={2023}
}
</code></pre>
</left></td></tr></table>

<!-- <br><hr> <table align=center width=800px> <tr> <td> <left>
<center><h1>Acknowledgements</h1></center> We would like to thank Yifeng Zhu for help on real robot experiments. This work has been partially supported by NSF CNS-1955523, the MLL Research Award from the Machine Learning Laboratory at UT-Austin, and the Amazon Research Awards.
 -->

<!-- </left></td></tr></table>
<br><br> -->

<div style="display:none">
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-PPXN40YS69"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-PPXN40YS69');
</script>
<!-- </center></div></body></div> -->

