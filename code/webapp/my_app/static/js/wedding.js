$(document).ready(function(){
  $('.jumbotron').css('background-color', 'transparent')

  opts = {
  lines: 11, // The number of lines to draw
  length: 0, // The length of each line
  width: 12, // The line thickness
  radius: 50, // The radius of the inner circle
  corners: 1, // Corner roundness (0..1)
  rotate: 1, // The rotation offset
  direction: 1, // 1: clockwise, -1: counterclockwise
  color: '#000', // #rgb or #rrggbb or array of colors
  speed: 0.9, // Rounds per second
  trail: 60, // Afterglow percentage
  shadow: false, // Whether to render a shadow
  hwaccel: false, // Whether to use hardware acceleration
  className: 'spinner', // The CSS class to assign to the spinner
  zIndex: 2e9, // The z-index (defaults to 2000000000)
  top: '50%', // Top position relative to parent
  left: '50%' // Left position relative to parent
};

  spinnerEl = $('.spinner')

  spinnerEl.on('click', function() {
    spinner = new Spinner(opts).spin()
    spinnerEl.append(spinner.el)
  })
})