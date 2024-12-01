import unittest
from bnetbase import Variable, Factor, BN
from naive_bayes import restrict, sum_out, multiply, normalize, ve

class TestRestrict(unittest.TestCase):
    def test_restrict_1(self):
        v1 = Variable('v1', ['T', 'F'])
        v2 = Variable('v2', ['T', 'F'])
        v3 = Variable('v3', ['T', 'F'])

        f = Factor('f', [v1, v2, v3])
        f.add_values([['T', 'T', 'T', 0.1],
                    ['T', 'T', 'F', 0.2],
                    ['T', 'F', 'T', 0.3],
                    ['T', 'F', 'F', 0.4],
                    ['F', 'T', 'T', 0.5],
                    ['F', 'T', 'F', 0.6],
                    ['F', 'F', 'T', 0.7],
                    ['F', 'F', 'F', 0.8]])

        res = restrict(f, v1, 'F')

        self.assertAlmostEqual(res.get_value(['T', 'T']), 0.5)
        self.assertAlmostEqual(res.get_value(['T', 'F']), 0.6)
        self.assertAlmostEqual(res.get_value(['F', 'T']), 0.7)
        self.assertAlmostEqual(res.get_value(['F', 'F']), 0.8)

class TestSumOut(unittest.TestCase):
    def test_sum_out_1(self):
        v1 = Variable('v1', ['T', 'F'])
        v2 = Variable('v2', ['T', 'F'])
        v3 = Variable('v3', ['T', 'F'])

        f = Factor('f', [v1, v2, v3])
        f.add_values([['T', 'T', 'T', 0.1],
                    ['T', 'T', 'F', 0.2],
                    ['T', 'F', 'T', 0.3],
                    ['T', 'F', 'F', 0.4],
                    ['F', 'T', 'T', 0.5],
                    ['F', 'T', 'F', 0.6],
                    ['F', 'F', 'T', 0.7],
                    ['F', 'F', 'F', 0.8]])

        res = sum_out(f, v2)

        self.assertAlmostEqual(res.get_value(['T', 'T']), 0.4)
        self.assertAlmostEqual(res.get_value(['T', 'F']), 0.6)
        self.assertAlmostEqual(res.get_value(['F', 'T']), 1.2)
        self.assertAlmostEqual(res.get_value(['F', 'F']), 1.4)

class TestNormalize(unittest.TestCase):
    def test_normalize_1(self):
        v = Variable("v", ["T", "F"])
        f = Factor("f", [v])
        f.add_values([["T", 1.2], ["F", 0.8]])

        f_norm = normalize(f)

        self.assertAlmostEqual(f_norm.get_value(["T"]), 0.6)
        self.assertAlmostEqual(f_norm.get_value(["F"]), 0.4)

class TestMultiply(unittest.TestCase):
    def test_multiply_1(self):
        v1 = Variable('v1', ['T', 'F'])
        v2 = Variable('v2', ['T', 'F'])
        v3 = Variable('v3', ['T', 'F'])

        f1 = Factor('f1', [v1, v3])
        f1.add_values([['T', 'T', 0.8],
                    ['F', 'T', 0.2],
                    ['T', 'F', 0.4],
                    ['F', 'F', 0.6]])

        f2 = Factor('f2', [v2, v3])
        f2.add_values([['T', 'T', 0.4],
                    ['F', 'T', 0.6],
                    ['T', 'F', 0.04],
                    ['F', 'F', 0.96]])

        res = multiply([f1, f2])
        assignments = [
            ['T', 'T', 'T'],
            ['T', 'T', 'F'],
            ['T', 'F', 'T'],
            ['T', 'F', 'F'],
            ['F', 'T', 'T'],
            ['F', 'T', 'F'],
            ['F', 'F', 'T'],
            ['F', 'F', 'F']
        ]

        expected_values = [0.32, 0.016, 0.08, 0.024, 0.48, 0.384, 0.12, 0.576]

        for assignment, expected in zip(assignments, expected_values):
            v2.set_assignment(assignment[0])
            v1.set_assignment(assignment[1])
            v3.set_assignment(assignment[2])
            self.assertAlmostEqual(res.get_value_at_current_assignments(), expected)
        
class TestVeSmall(unittest.TestCase):
    def test_ve_small_1(self):
        A = Variable('A', ['T', 'F'])
        W = Variable('W', ['T', 'F'])
        B = Variable('B', ['T', 'F'])

        f1 = Factor('f1', [B])
        f1.add_values([['T', 0.5], ['F', 0.5]])

        f2 = Factor('f2', [B, A])
        f2.add_values([['T', 'T', 0.3],
                    ['T', 'F', 0.7],
                    ['F', 'T', 0.4],
                    ['F', 'F', 0.6]])

        f3 = Factor('f3', [A, W])
        f3.add_values([['T', 'T', 0.1],
                    ['T', 'F', 0.9],
                    ['F', 'T', 0.2],
                    ['F', 'F', 0.8]])

        bn = BN("ve_small", [A, W, B], [f1, f2, f3])

        W.set_evidence('F')
        res = ve(bn, A, [W])
        self.assertAlmostEqual(res.get_value(['T']), 0.3772455089820359)
        self.assertAlmostEqual(res.get_value(['F']), 0.6227544910179641)

if __name__ == '__main__':
    unittest.main()
