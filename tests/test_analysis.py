import unittest
import numpy as np
from cycling_analysis.filtering import filter_trajectory
from cycling_analysis.cycles import detect_cycle_peaks, normalize_cycles
from cycling_analysis.kinematics import joint_angle, knee_flexion
from cycling_analysis.symmetry import cross_correlation
from cycling_analysis.stability import derivatives

class AnalysisMathTests(unittest.TestCase):
    def test_known_joint_angle(self):
        self.assertAlmostEqual(float(joint_angle(np.array([1.,0.,0.]), np.zeros(3), np.array([0.,1.,0.]))),90)
        pose=np.zeros((20,17,3)); pose[:,1]=[1,0,0]; pose[:,3]=[-1,0,0]
        np.testing.assert_allclose(knee_flexion(pose,1,2,3),0)

    def test_cross_correlation_lag(self):
        x=np.zeros(100); y=np.zeros(100); x[40]=1; y[35]=1
        result=cross_correlation(x,y)
        self.assertEqual(result['lag_frames'],5)
        self.assertGreater(result['coefficient'],.99)

    def test_cycles_sinusoid(self):
        fps=60; t=np.arange(600)/fps; signal=np.sin(2*np.pi*1.2*t)
        peaks=detect_cycle_peaks(signal,fps)
        self.assertTrue(10 <= len(peaks)-1 <= 12)
        self.assertEqual(normalize_cycles(signal,peaks,100).shape,(len(peaks)-1,100))

    def test_filter_reduces_high_frequency_noise(self):
        fps=60; t=np.arange(600)/fps; clean=np.sin(2*np.pi*t)
        noisy=clean+0.5*np.sin(2*np.pi*15*t)
        pose=np.tile(noisy[:,None,None],(1,17,3))
        filtered=filter_trajectory(pose,fps)
        self.assertEqual(filtered.shape,pose.shape)
        self.assertLess(np.sqrt(np.mean((filtered[:,0,0]-clean)**2)),.08)
        self.assertLess(np.std(filtered[:,0,0]-clean),np.std(noisy-clean)/3)

    def test_derivatives_units(self):
        t=np.arange(100)/50
        _,acceleration,jerk=derivatives(t*t,50)
        self.assertAlmostEqual(np.mean(acceleration[5:-5]),2,delta=.01)
        self.assertLess(np.mean(np.abs(jerk[5:-5])),.01)

if __name__=='__main__': unittest.main()
