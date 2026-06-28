package primitive

import (
	"image"
	"image/color"
	"math"
	"runtime"
	"testing"
	"time"
)

// syntheticTarget builds the same gradient+block target used by the Rust sweep, scaled to size×size
// (block kept at the same relative region), so the fogleman-Go vs GPU comparison is on identical content.
func syntheticTarget(size int) image.Image {
	w, h := size, size
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r := uint8(x * 255 / w)
			g := uint8(y * 255 / h)
			b := uint8((x + y) * 255 / (w + h))
			// x ∈ [w/4, 5w/8), y ∈ [5h/16, 11h/16) — == [16,40)×[20,44) at 64.
			if x*8 >= w*2 && x*8 < w*5 && y*16 >= h*5 && y*16 < h*11 {
				r, g, b = 230, 40, 90
			}
			img.Set(x, y, color.RGBA{r, g, b, 255})
		}
	}
	return img
}

func avgColor(img image.Image) Color {
	b := img.Bounds()
	var rs, gs, bs, n int64
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bb, _ := img.At(x, y).RGBA()
			rs += int64(r >> 8)
			gs += int64(g >> 8)
			bs += int64(bb >> 8)
			n++
		}
	}
	return Color{int(rs / n), int(gs / n), int(bs / n), 255}
}

// runShapes times `shapes` calls to Model.Step (triangle, alpha 128) at size×size with `workers`
// goroutines — the same n=1000/age=100/m=16 budget the GPU-3 run uses. Returns shapes/sec + PSNR.
func runShapes(workers, shapes, size int) (float64, float64) {
	target := syntheticTarget(size)
	model := NewModel(target, avgColor(target), size, workers)
	start := time.Now()
	for i := 0; i < shapes; i++ {
		model.Step(ShapeTypeTriangle, 128, 0)
	}
	sps := float64(shapes) / time.Since(start).Seconds()
	// PSNR from the running normalized-RMSE score: -20*log10(score), matching primitive-core.
	psnr := -20.0 * math.Log10(model.Score)
	return sps, psnr
}

func TestBenchFogleman(t *testing.T) {
	const shapes = 100
	cores := runtime.NumCPU()

	// Best of 3 (fastest = least interfered-with) per config, at a given canvas size.
	best := func(workers, size int) (float64, float64) {
		bestSps, psnr := 0.0, 0.0
		for r := 0; r < 3; r++ {
			s, p := runShapes(workers, shapes, size)
			if s > bestSps {
				bestSps, psnr = s, p
			}
		}
		return bestSps, psnr
	}

	for _, size := range []int{64, 128} {
		// Warm up (allocator, GC, page-ins) so timings reflect steady state.
		_, _ = runShapes(1, 5, size)
		_, _ = runShapes(cores, 5, size)

		s1, p1 := best(1, size)
		sN, pN := best(cores, size)

		t.Logf("FOGLEMAN-GO %dx%d triangle a=128, %d shapes (n=1000 age=100 m=16):", size, size, shapes)
		t.Logf("  -j1   (single core): %.1f shapes/s  (PSNR %.2f dB)", s1, p1)
		t.Logf("  -j%-2d  (all cores):   %.1f shapes/s  (PSNR %.2f dB)", cores, sN, pN)
	}
}
