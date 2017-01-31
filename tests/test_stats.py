import dev.sct_sdika as sdika
import msct_image as image

def test_compute_stats():
	img_true = image.Image()
	img_predicted = image.Image()
	img_segment = image.Image()
	sdika._compute_stats(img_true, img_predicted, img_segment)
