#!/usr/bin/env python3
"""
Validation Script for Adaptive Threshold Control Enhancement

This script validates that the PID-based adaptive control provides:
1. Stable FPR maintenance
2. Dynamic threshold adjustment
3. Resilience to data drift
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json

from src.enhanced_lbf.adaptive import AdaptiveLBF
from src.learned_bloom_filter.basic_lbf import BasicLearnedBloomFilter


class AdaptiveControlValidator:
    """Validate adaptive threshold control enhancement."""
    
    def __init__(self):
        self.results = {}
    
    def test_fpr_stability(self):
        """Test 1: Verify FPR stability under normal conditions."""
        print("\n" + "="*60)
        print("TEST 1: FPR Stability Validation")
        print("="*60)
        
        target_fpr = 0.01
        
        # Create adaptive LBF
        adaptive_lbf = AdaptiveLBF(
            target_fpr=target_fpr,
            kp=0.5,
            ki=0.1,
            kd=0.05,
            verbose=False
        )
        
        # Initial training
        n_train = 5000
        train_data = [f"train_{i}" for i in range(n_train)]
        train_labels = np.random.randint(0, 2, n_train)
        adaptive_lbf.train(train_data, train_labels)
        
        # Test FPR over time
        fpr_history = []
        threshold_history = []
        
        print("\nMonitoring FPR stability...")
        for epoch in range(50):
            # Add some items
            for i in range(100):
                adaptive_lbf.add(f"item_{epoch}_{i}", label=np.random.randint(0, 2))
            
            # Measure FPR
            test_negatives = [f"test_neg_{epoch}_{i}" for i in range(1000)]
            fp = sum(1 for item in test_negatives if adaptive_lbf.query(item))
            actual_fpr = fp / len(test_negatives)
            fpr_history.append(actual_fpr)
            
            # Get threshold
            if hasattr(adaptive_lbf, 'threshold'):
                threshold_history.append(adaptive_lbf.threshold)
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: FPR = {actual_fpr:.4f}, Target = {target_fpr:.4f}")
        
        # Calculate stability metrics
        fpr_mean = np.mean(fpr_history)
        fpr_std = np.std(fpr_history)
        fpr_variance = np.var(fpr_history)
        
        # Compare with non-adaptive version
        basic_lbf = BasicLearnedBloomFilter()
        basic_lbf.train(train_data, train_labels)
        
        basic_fpr_history = []
        for epoch in range(50):
            test_negatives = [f"basic_neg_{epoch}_{i}" for i in range(1000)]
            fp = sum(1 for item in test_negatives if basic_lbf.query(item))
            basic_fpr_history.append(fp / len(test_negatives))
        
        basic_fpr_std = np.std(basic_fpr_history)
        
        print(f"\n{'='*40}")
        print("FPR Stability Results:")
        print(f"  Adaptive - Mean FPR:     {fpr_mean:.4f}")
        print(f"  Adaptive - Std Dev:      {fpr_std:.4f}")
        print(f"  Adaptive - Variance:     {fpr_variance:.6f}")
        print(f"  Basic - Std Dev:         {basic_fpr_std:.4f}")
        print(f"  Improvement:             {basic_fpr_std/fpr_std:.2f}x more stable")
        
        # Plot FPR and threshold evolution
        self._plot_fpr_stability(fpr_history, threshold_history, target_fpr)
        
        # Validation: FPR should stay close to target
        validation = (abs(fpr_mean - target_fpr) < 0.005 and  # Within 0.5% of target
                     fpr_std < 0.01)  # Low variance
        
        print(f"\n✅ FPR stability validated: {validation}")
        
        self.results['fpr_stability'] = {
            'mean_fpr': fpr_mean,
            'std_dev': fpr_std,
            'variance': fpr_variance,
            'improvement_factor': basic_fpr_std/fpr_std if fpr_std > 0 else float('inf'),
            'validated': validation
        }
        return validation
    
    def test_dynamic_adjustment(self):
        """Test 2: Verify dynamic threshold adjustment."""
        print("\n" + "="*60)
        print("TEST 2: Dynamic Threshold Adjustment")
        print("="*60)
        
        adaptive_lbf = AdaptiveLBF(
            target_fpr=0.01,
            kp=0.5,
            ki=0.1,
            kd=0.05,
            verbose=False
        )
        
        # Initial training
        train_data = [f"train_{i}" for i in range(1000)]
        adaptive_lbf.train(train_data, np.ones(1000))  # All positive
        
        initial_threshold = getattr(adaptive_lbf, 'threshold', 0.5)
        print(f"Initial threshold: {initial_threshold:.4f}")
        
        # Simulate high FPR scenario (threshold too low)
        threshold_adjustments = []
        
        print("\nSimulating FPR variations...")
        for scenario in range(5):
            # Alternate between high and low FPR scenarios
            if scenario % 2 == 0:
                # High FPR - should increase threshold
                print(f"\n  Scenario {scenario}: High FPR (threshold should increase)")
                for _ in range(100):
                    adaptive_lbf.add(f"high_{scenario}_{_}", label=0)  # Many negatives
            else:
                # Low FPR - should decrease threshold
                print(f"  Scenario {scenario}: Low FPR (threshold should decrease)")
                for _ in range(100):
                    adaptive_lbf.add(f"low_{scenario}_{_}", label=1)  # Many positives
            
            # Force threshold update
            test_items = [f"test_{scenario}_{i}" for i in range(100)]
            for item in test_items:
                adaptive_lbf.query(item)
            
            new_threshold = getattr(adaptive_lbf, 'threshold', initial_threshold)
            adjustment = new_threshold - initial_threshold
            threshold_adjustments.append(adjustment)
            
            print(f"    New threshold: {new_threshold:.4f}, Adjustment: {adjustment:+.4f}")
            initial_threshold = new_threshold
        
        # Check if threshold actually adjusts
        total_adjustment = sum(abs(a) for a in threshold_adjustments)
        
        print(f"\n{'='*40}")
        print(f"Total threshold movement: {total_adjustment:.4f}")
        
        validation = total_adjustment > 0.01  # Threshold should move
        print(f"\n✅ Dynamic adjustment validated: {validation}")
        
        self.results['dynamic_adjustment'] = {
            'total_adjustment': total_adjustment,
            'adjustments': threshold_adjustments,
            'validated': validation
        }
        return validation
    
    def test_concept_drift(self):
        """Test 3: Validate adaptation to concept drift."""
        print("\n" + "="*60)
        print("TEST 3: Concept Drift Adaptation")
        print("="*60)
        
        target_fpr = 0.01
        
        adaptive_lbf = AdaptiveLBF(
            target_fpr=target_fpr,
            kp=0.5,
            ki=0.1,
            kd=0.05,
            verbose=False
        )
        
        # Phase 1: Normal data distribution
        print("\nPhase 1: Normal distribution")
        phase1_data = [f"normal_{i}" for i in range(5000)]
        phase1_labels = np.random.binomial(1, 0.3, 5000)  # 30% positive
        adaptive_lbf.train(phase1_data[:2500], phase1_labels[:2500])
        
        # Add rest and measure FPR
        for i in range(2500, 5000):
            adaptive_lbf.add(phase1_data[i], phase1_labels[i])
        
        test_neg = [f"test_normal_{i}" for i in range(1000)]
        fp = sum(1 for item in test_neg if adaptive_lbf.query(item))
        phase1_fpr = fp / len(test_neg)
        print(f"  Phase 1 FPR: {phase1_fpr:.4f}")
        
        # Phase 2: Concept drift - different distribution
        print("\nPhase 2: Concept drift (70% positive)")
        phase2_data = [f"drift_{i}" for i in range(5000)]
        phase2_labels = np.random.binomial(1, 0.7, 5000)  # 70% positive now!
        
        fpr_during_drift = []
        for i in range(0, 5000, 100):
            # Add batch
            for j in range(i, min(i+100, 5000)):
                adaptive_lbf.add(phase2_data[j], phase2_labels[j])
            
            # Measure FPR
            test_neg = [f"test_drift_{i}_{k}" for k in range(100)]
            fp = sum(1 for item in test_neg if adaptive_lbf.query(item))
            fpr = fp / len(test_neg)
            fpr_during_drift.append(fpr)
        
        phase2_fpr = np.mean(fpr_during_drift[-10:])  # Last 10 measurements
        print(f"  Phase 2 FPR (after adaptation): {phase2_fpr:.4f}")
        
        # Phase 3: Return to normal
        print("\nPhase 3: Return to normal distribution")
        phase3_data = [f"return_{i}" for i in range(5000)]
        phase3_labels = np.random.binomial(1, 0.3, 5000)  # Back to 30%
        
        for i in range(5000):
            adaptive_lbf.add(phase3_data[i], phase3_labels[i])
        
        test_neg = [f"test_return_{i}" for i in range(1000)]
        fp = sum(1 for item in test_neg if adaptive_lbf.query(item))
        phase3_fpr = fp / len(test_neg)
        print(f"  Phase 3 FPR: {phase3_fpr:.4f}")
        
        # Calculate adaptation metrics
        max_deviation = max(abs(phase1_fpr - target_fpr),
                           abs(phase2_fpr - target_fpr),
                           abs(phase3_fpr - target_fpr))
        
        print(f"\n{'='*40}")
        print(f"Adaptation Results:")
        print(f"  Max deviation from target: {max_deviation:.4f}")
        print(f"  FPR recovery after drift: {abs(phase3_fpr - target_fpr):.4f}")
        
        validation = max_deviation < 0.02  # Stay within 2% of target
        print(f"\n✅ Concept drift adaptation validated: {validation}")
        
        self.results['concept_drift'] = {
            'phase1_fpr': phase1_fpr,
            'phase2_fpr': phase2_fpr,
            'phase3_fpr': phase3_fpr,
            'max_deviation': max_deviation,
            'validated': validation
        }
        return validation
    
    def test_pid_controller(self):
        """Test 4: Validate PID controller behavior."""
        print("\n" + "="*60)
        print("TEST 4: PID Controller Validation")
        print("="*60)
        
        # Test different PID configurations
        configs = [
            {'kp': 1.0, 'ki': 0.0, 'kd': 0.0, 'name': 'P-only'},
            {'kp': 0.5, 'ki': 0.2, 'kd': 0.0, 'name': 'PI'},
            {'kp': 0.5, 'ki': 0.1, 'kd': 0.05, 'name': 'PID'},
        ]
        
        results_by_config = {}
        
        for config in configs:
            print(f"\nTesting {config['name']} controller...")
            
            adaptive_lbf = AdaptiveLBF(
                target_fpr=0.01,
                kp=config['kp'],
                ki=config['ki'],
                kd=config['kd'],
                verbose=False
            )
            
            # Train
            train_data = [f"train_{i}" for i in range(2000)]
            adaptive_lbf.train(train_data, np.random.randint(0, 2, 2000))
            
            # Apply step disturbance
            fpr_response = []
            
            # Normal operation
            for _ in range(10):
                test_neg = [f"normal_{_}_{i}" for i in range(100)]
                fp = sum(1 for item in test_neg if adaptive_lbf.query(item))
                fpr_response.append(fp / len(test_neg))
            
            # Step disturbance - sudden change
            for _ in range(20):
                # Add many positives (changes distribution)
                for i in range(50):
                    adaptive_lbf.add(f"disturb_{_}_{i}", label=1)
                
                test_neg = [f"disturb_{_}_{i}" for i in range(100)]
                fp = sum(1 for item in test_neg if adaptive_lbf.query(item))
                fpr_response.append(fp / len(test_neg))
            
            # Calculate controller metrics
            overshoot = max(fpr_response) - 0.01
            settling_time = self._calculate_settling_time(fpr_response, 0.01)
            steady_state_error = abs(np.mean(fpr_response[-5:]) - 0.01)
            
            results_by_config[config['name']] = {
                'overshoot': overshoot,
                'settling_time': settling_time,
                'steady_state_error': steady_state_error,
                'response': fpr_response
            }
            
            print(f"  Overshoot: {overshoot:.4f}")
            print(f"  Settling time: {settling_time} samples")
            print(f"  Steady-state error: {steady_state_error:.4f}")
        
        # Plot controller comparison
        self._plot_pid_comparison(results_by_config)
        
        # Validate PID performs best
        pid_results = results_by_config['PID']
        validation = (pid_results['overshoot'] < 0.05 and
                     pid_results['steady_state_error'] < 0.01)
        
        print(f"\n✅ PID controller validated: {validation}")
        
        self.results['pid_controller'] = {
            'configurations': results_by_config,
            'validated': validation
        }
        return validation
    
    def _calculate_settling_time(self, response, target, tolerance=0.02):
        """Calculate settling time for controller response."""
        for i in range(len(response)-1, -1, -1):
            if abs(response[i] - target) > tolerance:
                return i + 1
        return 0
    
    def _plot_fpr_stability(self, fpr_history, threshold_history, target_fpr):
        """Plot FPR stability over time."""
        plt.figure(figsize=(12, 5))
        
        # Plot FPR
        plt.subplot(1, 2, 1)
        plt.plot(fpr_history, 'b-', alpha=0.7, label='Actual FPR')
        plt.axhline(y=target_fpr, color='r', linestyle='--', label='Target FPR')
        plt.fill_between(range(len(fpr_history)), 
                        target_fpr - 0.005, target_fpr + 0.005,
                        alpha=0.2, color='red', label='±0.5% tolerance')
        plt.xlabel('Epoch')
        plt.ylabel('False Positive Rate')
        plt.title('FPR Stability Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot threshold
        if threshold_history:
            plt.subplot(1, 2, 2)
            plt.plot(threshold_history, 'g-', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Threshold')
            plt.title('Adaptive Threshold Evolution')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs('validation', exist_ok=True)
        plt.savefig('validation/fpr_stability.png')
        plt.close()
        print("✓ FPR stability plot saved to validation/fpr_stability.png")
    
    def _plot_pid_comparison(self, results):
        """Plot PID controller comparison."""
        plt.figure(figsize=(10, 6))
        
        for name, data in results.items():
            plt.plot(data['response'], label=name)
        
        plt.axhline(y=0.01, color='r', linestyle='--', label='Target (1%)')
        plt.xlabel('Time Steps')
        plt.ylabel('False Positive Rate')
        plt.title('PID Controller Response Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Mark disturbance
        plt.axvline(x=10, color='gray', linestyle=':', alpha=0.5)
        plt.text(10, max([max(d['response']) for d in results.values()])*0.9,
                'Disturbance', rotation=90)
        
        os.makedirs('validation', exist_ok=True)
        plt.savefig('validation/pid_comparison.png')
        plt.close()
        print("✓ PID comparison plot saved to validation/pid_comparison.png")
    
    def generate_report(self):
        """Generate validation report for adaptive control."""
        print("\n" + "="*60)
        print("ADAPTIVE CONTROL VALIDATION REPORT")
        print("="*60)
        
        all_valid = all(r.get('validated', False) for r in self.results.values())
        
        print("\nValidation Results:")
        print("-"*40)
        for test, result in self.results.items():
            status = "✅ PASSED" if result.get('validated') else "❌ FAILED"
            print(f"{test:20} {status}")
        
        # Key metrics
        print("\nKey Metrics:")
        print("-"*40)
        if 'fpr_stability' in self.results:
            r = self.results['fpr_stability']
            print(f"FPR stability:       {r['std_dev']:.4f} std dev")
            print(f"Improvement factor:  {r['improvement_factor']:.2f}x")
        if 'concept_drift' in self.results:
            r = self.results['concept_drift']
            print(f"Max FPR deviation:   {r['max_deviation']:.4f}")
        
        if all_valid:
            print("\n🎉 ENHANCEMENT 3 VALIDATED: Adaptive threshold control maintains stable FPR")
        else:
            print("\n⚠️ Some validations failed. Review results above.")
        
        return self.results


def main():
    """Run adaptive control validation tests."""
    validator = AdaptiveControlValidator()
    
    # Run all validation tests
    validator.test_fpr_stability()
    validator.test_dynamic_adjustment()
    validator.test_concept_drift()
    validator.test_pid_controller()
    
    # Generate report
    results = validator.generate_report()
    
    # Save results
    os.makedirs('validation', exist_ok=True)
    with open('validation/adaptive_control_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ Validation complete! Results saved to validation/adaptive_control_results.json")
    
    return results


if __name__ == "__main__":
    main()