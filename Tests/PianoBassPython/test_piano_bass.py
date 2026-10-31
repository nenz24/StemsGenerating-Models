"""Harmony-template and real SoundFont-rendering checks for the CPU prototype."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from piano_bass import CHORD_TYPES, chord_templates, render
from bass_performance import arrange


class PianoBassTests(unittest.TestCase):
    def test_performance_preserves_silence_and_monophonic_timing(self):
        # Beat 10 is intentionally silent; all others share a held harmony.
        report={'duration_seconds':24.,'chords':[
            {'start':float(i),'end':float(i+1),'chord':'E'} for i in range(24)],
            'notes':[{'start':float(i),'end':i+.84,'midi':40,'velocity':74}
                     for i in range(24) if i not in [1,10]]}
        notes=arrange(report)
        self.assertTrue(all(28<=n['midi']<=48 and 0<=n['start']<n['end']<=24 for n in notes))
        self.assertTrue(all(a['end']<=b['start'] for a,b in zip(notes,notes[1:])))
        self.assertFalse(any(n['start']<11 and n['end']>10 for n in notes))
        self.assertFalse(any(n['start']<2 and n['end']>1 for n in notes))
        self.assertGreater(len({round(n['end']-n['start'],2) for n in notes}),3)

    def test_transposed_chords_with_a_loud_third_keep_their_root(self):
        templates = chord_templates()
        for root in range(12):
            for kind, (_, intervals) in enumerate(CHORD_TYPES[:2]):
                chroma = np.zeros(12)
                chroma[(root+np.array(intervals))%12] = [0.7, 1.0, 0.8]
                self.assertEqual(int(np.argmax(templates@chroma)),kind*12+root)

    def test_real_renderer_preserves_time_and_does_not_clip(self):
        soundfont = Path('/usr/share/sounds/sf2/TimGM6mb.sf2')
        if not soundfont.is_file():
            self.skipTest('GM SoundFont is not installed')
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'bass.wav'
            render([{'start':.25,'end':1.2,'midi':40,'velocity':80}],2.,target,soundfont)
            audio,sr=sf.read(target)
            self.assertEqual((sr,audio.shape),(44100,(88200,2)))
            self.assertTrue(np.isfinite(audio).all())
            self.assertLess(np.max(np.abs(audio[:10000])),1e-6)
            self.assertAlmostEqual(float(np.max(np.abs(audio))),.7,places=5)
            # E2 fundamental must be present in the rendered electric bass.
            mono=audio[16000:40000].mean(axis=1)
            spectrum=np.abs(np.fft.rfft(mono*np.hanning(len(mono))))
            freq=np.fft.rfftfreq(len(mono),1/sr)
            peak=freq[np.argmax(spectrum*(freq>60)*(freq<110))]
            self.assertLess(abs(peak-82.4069),3.)


if __name__=='__main__':
    unittest.main()
