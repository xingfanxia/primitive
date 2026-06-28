package primitive

// Parity fixture generator. Dumps the exact outputs of fogleman's unexported pixel math
// (rasterizeTriangle, cropScanlines, computeColor, copyLines+drawLines, differenceFull,
// differencePartial) on fixed inputs, so the Rust port (primitive-core) can assert
// bit-for-bit parity without depending on Go at CI time.
//
// Run:  go test -run TestDumpParityFixture ./primitive
// Writes: ../../crates/primitive-core/tests/fixtures/parity_fogleman.json

import (
	"encoding/json"
	"image"
	"os"
	"testing"
)

// Deterministic synthetic target so both languages build an identical image.
func buildTarget(w, h int) *image.RGBA {
	im := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := im.PixOffset(x, y)
			im.Pix[i+0] = uint8((x*7 + y*13) & 255)
			im.Pix[i+1] = uint8((x*3 + y*5) & 255)
			im.Pix[i+2] = uint8((x + y*2) & 255)
			im.Pix[i+3] = 255
		}
	}
	return im
}

func solid(w, h int, r, g, b uint8) *image.RGBA {
	im := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < len(im.Pix); i += 4 {
		im.Pix[i+0] = r
		im.Pix[i+1] = g
		im.Pix[i+2] = b
		im.Pix[i+3] = 255
	}
	return im
}

func sseInt(a, b *image.RGBA) uint64 {
	var total uint64
	for i := 0; i < len(a.Pix); i += 4 {
		for c := 0; c < 4; c++ {
			d := int(a.Pix[i+c]) - int(b.Pix[i+c])
			total += uint64(d * d)
		}
	}
	return total
}

type fixtureCase struct {
	Tri               [6]int     `json:"tri"`
	Scanlines         [][4]int64 `json:"scanlines"` // y, x1, x2, alpha
	Color             [4]int     `json:"color"`
	Covered           [][5]int   `json:"covered"` // offset, r, g, b... we store r,g,b,a
	SSEAfter          uint64     `json:"sse_after"`
	DifferenceFullAft float64    `json:"difference_full_after"`
	DifferencePartial float64    `json:"difference_partial"`
}

type fixture struct {
	Width                 int           `json:"width"`
	Height                int           `json:"height"`
	Current               [4]int        `json:"current"`
	Alpha                 int           `json:"alpha"`
	DifferenceFullCurrent float64       `json:"difference_full_current"`
	Cases                 []fixtureCase `json:"cases"`
}

func TestDumpParityFixture(t *testing.T) {
	const w, h = 40, 30
	const alpha = 128
	target := buildTarget(w, h)
	current := solid(w, h, 100, 110, 120)

	scoreCurrent := differenceFull(target, current)

	tris := [][6]int{
		{5, 5, 30, 8, 15, 25},
		{-3, -3, 20, 2, 8, 22},
		{0, 0, 39, 0, 39, 29},
		{18, 4, 35, 20, 6, 26},
	}

	fx := fixture{
		Width:                 w,
		Height:                h,
		Current:               [4]int{100, 110, 120, 255},
		Alpha:                 alpha,
		DifferenceFullCurrent: scoreCurrent,
	}

	for _, tri := range tris {
		var buf []Scanline
		raw := rasterizeTriangle(tri[0], tri[1], tri[2], tri[3], tri[4], tri[5], buf)
		lines := cropScanlines(raw, w, h)

		color := computeColor(target, current, lines, alpha)

		// after = current with the shape composited (Worker.Energy uses a line-local buffer;
		// compositing a full copy gives the same covered pixels and lets us score the whole image).
		after := image.NewRGBA(image.Rect(0, 0, w, h))
		copy(after.Pix, current.Pix)
		copyLines(after, current, lines)
		drawLines(after, color, lines)

		var sl [][4]int64
		var covered [][5]int
		for _, ln := range lines {
			sl = append(sl, [4]int64{int64(ln.Y), int64(ln.X1), int64(ln.X2), int64(ln.Alpha)})
			i := after.PixOffset(ln.X1, ln.Y)
			for x := ln.X1; x <= ln.X2; x++ {
				covered = append(covered, [5]int{i,
					int(after.Pix[i+0]), int(after.Pix[i+1]), int(after.Pix[i+2]), int(after.Pix[i+3])})
				i += 4
			}
		}

		fx.Cases = append(fx.Cases, fixtureCase{
			Tri:               tri,
			Scanlines:         sl,
			Color:             [4]int{color.R, color.G, color.B, color.A},
			Covered:           covered,
			SSEAfter:          sseInt(target, after),
			DifferenceFullAft: differenceFull(target, after),
			DifferencePartial: differencePartial(target, current, after, scoreCurrent, lines),
		})
	}

	out, err := json.MarshalIndent(fx, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	path := "../../crates/primitive-core/tests/fixtures/parity_fogleman.json"
	if err := os.MkdirAll("../../crates/primitive-core/tests/fixtures", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %s (%d cases)", path, len(fx.Cases))
}
