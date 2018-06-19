package nbnet

import (
	"os"
	"fmt"
	"image/png"
	"image"
	"bytes"
	"encoding/base64"
)

// predicts form image, assumes image in the form 28x28 png
func PredictFromImage(net Network, path string) int {
	input := pixelData(path)
	output := net.Predict(input)
	printMatrix(output)

	best := 0
	highest := 0.0
	for i := 0; i < net.outputs; i++ {
		if output.At(i, 0) > highest {
			best = i
			highest = output.At(i, 0)
		}
	}
	return best
}

// get pixel data form image
func pixelData(filePath string) (pixels []float64) {
	// read file
	file, err := os.Open(filePath)
	defer file.Close()
	if err != nil {
		fmt.Println("cannot read file")
	}

	img, err := png.Decode(file)
	if err != nil {
		fmt.Println("cannot decode file")
	}

	// create grayscale image
	bounds := img.Bounds()
	gray := image.NewGray(bounds)
	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			rgba := img.At(x, y)
			gray.Set(x, y, rgba)
		}
	}

	// create the pixel array
	pixels = make([]float64, len(gray.Pix))
	// subtract pix from 255 to get grayscale image inverse
	for i := 0; i < len(gray.Pix); i++ {
		pixels[i] = (float64(255-gray.Pix[i]) / 255.0 * 0.999) + 0.001
	}
	return
}

func GetImage(filePath string) image.Image {
	file, err := os.Open(filePath)
	defer file.Close()
	if err != nil {
		fmt.Println("unable to open file ", err)
	}

	img, _, err := image.Decode(file)
	if err != nil {
		fmt.Println("unable to decode file", err)
	}
	return img
}

func PrintImage(img image.Image) {
	var buf bytes.Buffer
	png.Encode(&buf, img)
	imgBase64Str := base64.StdEncoding.EncodeToString(buf.Bytes())
	fmt.Printf("\x1b]1337;File=inline=1:%s\a\n", imgBase64Str)
}