$('.flex-images').flexImages({rowHeight: 170});

$(".item").click(function() {
  $("#overlay").fadeIn();
  $("#overlay").append(
    $("<img>", {"src": $(this).children("img").attr("src")})
  );
});
$("#overlay").click(function() {
  $("#overlay").fadeOut();
  $("#overlay").children().remove();
});
