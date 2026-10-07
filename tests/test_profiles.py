import unittest
from pi_hunyuan.profiles import PROFILES, plan, normalize


class ProfilesTests(unittest.TestCase):
    def test_every_card_has_headroom_and_no_forced_resolution(self):
        for card, recipe in PROFILES.items():
            with self.subTest(card=card):
                actual = plan(float(card), card)
                self.assertLess(actual['budget_gib'], float(card))
                self.assertGreaterEqual(actual['headroom_gib'], 1.5)
                self.assertEqual(actual['tiled'], int(card) <= 12)
                self.assertNotIn('width', actual)

    def test_smaller_profiles_are_safe_simulations_on_32g(self):
        for card in ('8','10','12','16','24'):
            with self.subTest(card=card):
                profile = plan(31.8, card)
                self.assertTrue(profile['simulation'])
                self.assertEqual(profile['budget_gib'], PROFILES[card]['budget_gib'])

    def test_auto_selects_nearest_card_and_mode(self):
        self.assertEqual(plan(31.8)['nominal_gb'], 32)
        self.assertEqual(plan(7.8)['nominal_gb'], 8)
        self.assertEqual(plan(31.8, mode='speed')['headroom_gib'], 0)
        self.assertLessEqual(plan(31.8, mode='low-memory')['budget_gib'], 8)
        self.assertLessEqual(plan(31.8, '8', mode='speed')['budget_gib'], 6)

    def test_larger_profile_does_not_exceed_real_card(self):
        self.assertLessEqual(plan(8, '32')['budget_gib'], 6.5)
        self.assertEqual(normalize('12 GB'), '12')
        for value in ('48', 'banana'):
            with self.assertRaises(ValueError):
                normalize(value)
