import torch
import unittest
from src.train.trainer import ThreeTierCausalLoss

class TestThreeTierLoss(unittest.TestCase):
    def test_descendant_identification(self):
        """
        Test if get_descendants correctly identifies downstream nodes.
        Graph: 0 -> 1 -> 2. Node 3 is independent.
        Intervention on 0.
        Expected Descendants: 1, 2.
        """
        loss_fn = ThreeTierCausalLoss()
        
        # Adjacency Matrix (4 nodes)
        # 0->1, 1->2
        adj = torch.tensor([
            [0, 1, 0, 0], # 0
            [0, 0, 1, 0], # 1
            [0, 0, 0, 0], # 2
            [0, 0, 0, 0]  # 3
        ], dtype=torch.float32)
        
        # Mask: Intervene on 0
        mask = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32) # Batch size 1
        
        descendants = loss_fn.get_descendants(adj, mask)
        
        # Expected: 1 and 2 are descendants
        expected = torch.tensor([[0, 1, 1, 0]], dtype=torch.float32)
        
        print(f"Adj:\n{adj}")
        print(f"Mask:\n{mask}")
        print(f"Descendants:\n{descendants}")
        
        self.assertTrue(torch.equal(descendants, expected), "Descendant identification failed.")

    def test_weighting_logic(self):
        """
        Test if weights are applied correctly.
        Graph: 0 -> 1. 2 is noise.
        Intervention on 0.
        
        Weights:
        - Node 0 (Cause): 10.0
        - Node 1 (Effect): 5.0
        - Node 2 (Noise): 1.0
        
        We set errors to 1.0 for all nodes.
        Expected Loss = (10*1 + 5*1 + 1*1) / 3 = 16/3 = 5.333
        """
        loss_fn = ThreeTierCausalLoss(w_cause=10.0, w_effect=5.0, w_noise=1.0)
        
        adj = torch.tensor([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=torch.float32)
        
        mask = torch.tensor([[1, 0, 0]], dtype=torch.float32)
        
        # Pred = 1, Target = 0 -> Error = 1 (Huber delta=1)
        pred = torch.ones(1, 3)
        target = torch.zeros(1, 3)
        
        loss = loss_fn(pred, target, mask, adj)
        
        expected_loss = (10.0 * 0.5 + 5.0 * 0.5 + 1.0 * 0.5) / 3 # Huber(1.0) = 0.5 * 1^2 = 0.5
        
        print(f"Calculated Loss: {loss.item()}")
        print(f"Expected Loss: {expected_loss}")
        
        self.assertAlmostEqual(loss.item(), expected_loss, places=4, msg="Weighting logic failed.")

if __name__ == '__main__':
    unittest.main()
