$(document).ready(function() {
  // Handle form submission
  $('form').submit(function(event) {
    event.preventDefault();  // Prevent default form submission

    var formData = new FormData(this);

    // Make AJAX request to upload the file
    $.ajax({
      url: '/api/upload',
      type: 'POST',
      data: formData,
      contentType: false,
      processData: false,
      success: function(response) {
        // Retrieve the image data from the response
        var imageData = response.image;

        // Create an image element and set the source as the image data
        var img = document.createElement('img');
        img.src = 'data:image/png;base64,' + imageData;

        // Append the image element to a container on the page
        $('#image-container').empty().append(img);
      },
      error: function(error) {
        console.log('Error:', error);
      }
    });
  });

  // Handle pie chart click
  $('#piechart').click(function() {
    var imagePath = 'static/images/pie.png';
    var img = document.createElement('img');
    img.src = imagePath;
    
    // Replace the image in the image container
    $('#image-container').empty().append(img);
  });
});

function downloadFiles() {
  var link1 = document.getElementById('download-link-1');
  var link2 = document.getElementById('download-link-2');
  
  link1.click();
  link2.click();
}

function enableDownloadLink() {
  var downloadLink = document.getElementById('download-link');
  downloadLink.removeAttribute('disabled');
}
