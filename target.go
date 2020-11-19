package goertzel

import (
	"encoding/binary"
	"math"
	"sync"
	"time"
)

// Target is a target frequency detector.  It is a low-level tool which
// implements the Goertzel algorithm to detect the presence of a frequency on a
// block-wise basis.
type Target struct {

	//
	// Constants
	//

	// UseOptimized indicates that an optimized (phase-insensitive) Goertzel should be used for faster arithmetic
	UseOptimized bool

	// Frequency in Hz
	Frequency float64

	// Threshold is the threshold at which this frequency is determined to be present
	Threshold float64

	// sampleRate is the number of times per second that we should receive a sample
	sampleRate float64

	// channelNum specifies the number of channels, 1 = mono, 2 = stereo
	channelNum int

	// bitDepthInBytes argument specifies the number of bytes per sample per channel. The usual value
	// is 2. Only values 1 and 2 are supported
	bitDepthInBytes int

	sin       float64
	cos       float64
	coeff     float64
	blockSize int

	//
	//  Working Variables
	//

	realM float64
	imagM float64

	// Magnitude2 is the square of the magnitude of the last-processed block
	Magnitude2 float64

	buf []byte

	// blockReader variables for managing output of block summaries
	blockReaderPresent bool
	blockReader        chan *BlockSummary
	blockReaderMu      sync.Mutex

	stopped bool

	mu sync.Mutex
}

// NewTarget creates a Goertzel processor tuned to the given frequency
func NewTarget(freq, sampleRate float64, channelNum int, depthInBytes int, minDuration time.Duration) *Target {
	t := &Target{
		Frequency:       freq,
		sampleRate:      sampleRate,
		blockSize:       optimalBlockSize2(freq, sampleRate, minDuration),
		Threshold:       ToneThreshold,
		channelNum:      channelNum,
		bitDepthInBytes: depthInBytes,
		buf:             make([]byte, 0),
	}
	t.generateConstants()

	go func() {
		t.ingest()
	}()

	return t
}

// SetBlockSize overrides automatic calculation of the optimal N (block size) value and uses the one provided instead
func (t *Target) SetBlockSize(n int) {
	t.mu.Lock()
	t.blockSize = n
	t.mu.Unlock()

	t.generateConstants()
}

func (t *Target) generateConstants() {
	t.mu.Lock()
	defer t.mu.Unlock()

	N := float64(t.blockSize)
	rate := t.sampleRate

	k := math.Floor(0.5 + (N*t.Frequency)/rate)
	w := (2.0 * math.Pi / N) * float64(k)
	t.cos = math.Cos(w)
	t.sin = math.Sin(w)
	t.coeff = 2.0 * t.cos
}

func (t *Target) Write(p []byte) (int, error) {
	lenP := len(p)
	lenBuf := len(t.buf)
	buf := make([]byte, lenBuf+lenP)
	copy(buf[:lenBuf], t.buf)
	n := copy(buf[lenBuf:], p)
	t.buf = buf

	return n, nil
}

func (t *Target) ingest() {
	var i int
	var sample float64
	var q, q1, q2 float64

	defer t.Stop()

	for {
		if len(t.buf) < t.channelNum*t.bitDepthInBytes {
			continue
		}

		var buf = make([]byte, t.channelNum*t.bitDepthInBytes)

		copy(buf, t.buf[:t.channelNum*t.bitDepthInBytes])
		t.buf = t.buf[t.channelNum*t.bitDepthInBytes:]

		if t.channelNum == 2 {
			sampleL := int16(binary.LittleEndian.Uint16(buf[:t.bitDepthInBytes]))
			sampleR := int16(binary.LittleEndian.Uint16(buf[t.bitDepthInBytes:]))
			sample = float64(sampleL)*0.5 + float64(sampleR)*0.5
		} else {
			sample = float64(int16(binary.LittleEndian.Uint16(buf[:t.bitDepthInBytes])))
		}

		i++
		q = t.coeff*q1 - q2 + sample

		q2 = q1
		q1 = q

		if i == t.blockSize {
			t.calculateMagnitude(q1, q2)
			t.sendBlockSummary()
			i = 0
			q1 = 0
			q2 = 0

			if t.stopped {
				return
			}
		}
	}
}

func (t *Target) calculateMagnitude(q1, q2 float64) {
	if t.UseOptimized {
		t.Magnitude2 = q1*q1 + q2*q2 - q1*q2*t.coeff
		return
	}

	var scalingFactor = float64(t.blockSize) / 2.0

	t.mu.Lock()
	t.realM = (q1 - q2*t.cos) / scalingFactor
	t.imagM = (q2 * t.sin) / scalingFactor
	t.Magnitude2 = t.realM*t.realM + t.imagM*t.imagM
	t.mu.Unlock()
}

func (t *Target) sendBlockSummary() {
	if t.blockReaderPresent {
		select {
		case t.blockReader <- t.blockSummary():
		default:
		}
	}
}

func (t *Target) blockSummary() *BlockSummary {
	return &BlockSummary{
		Magnitude2: t.Magnitude2,
		Frequency:  t.Frequency,
		Duration:   time.Duration(float64(t.blockSize)/t.sampleRate) * time.Second,
		Samples:    t.blockSize,
		Present:    t.Magnitude2 > t.Threshold,
	}
}

func (t *Target) Close() error {
	t.Stop()
	return nil
}

// Stop terminates the Target processing.  It will close the Events channel and stop processing new data.
func (t *Target) Stop() {
	t.blockReaderMu.Lock()
	if t.blockReaderPresent {
		close(t.blockReader)
		t.blockReader = nil
		t.blockReaderPresent = false
	}
	t.blockReaderMu.Unlock()

	t.stopped = true
}

// Blocks returns a channel over which the summary of each resulting block from
// the Target frequency processor will be returned.  If Blocks() has already
// been called, nil will be returned.
func (t *Target) Blocks() <-chan *BlockSummary {
	t.blockReaderMu.Lock()
	if t.blockReaderPresent {
		t.blockReaderMu.Unlock()
		return nil
	}

	t.blockReaderPresent = true
	t.blockReader = make(chan *BlockSummary, BlockBufferSize)
	t.blockReaderMu.Unlock()

	return t.blockReader
}

// BlockSummary describes the result of a single block of processing for a Target frequency
type BlockSummary struct {

	// Magnitude2 is the square of the relative magnitude of the frequency in this block
	Magnitude2 float64

	// Frequency is the frequency which was being detected
	Frequency float64

	// Duration is the elapsed time which this block represents
	Duration time.Duration

	// Samples is the number of samples this block represents
	Samples int

	// Present indicates whether the frequency was found in the block, as determined by the target's threshold
	Present bool
}
