import unittest

import app


class CropRecommendationAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_home_and_health_endpoints(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_valid_inputs_return_a_crop_recommendation(self):
        response = self.client.post(
            "/predict",
            data={
                "Nitrogen": "90",
                "Phosphorus": "42",
                "Potassium": "43",
                "Temperature": "25",
                "Humidity": "80",
                "pH": "6.5",
                "Rainfall": "200",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"muskmelon", response.data.lower())

    def test_invalid_inputs_return_a_clear_error(self):
        response = self.client.post(
            "/predict",
            data={"Nitrogen": "not-a-number"},
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"nitrogen must be a valid number", response.data.lower())

    def test_non_finite_inputs_are_rejected(self):
        response = self.client.post(
            "/predict",
            data={"Nitrogen": "nan"},
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"nitrogen must be a finite number", response.data.lower())


if __name__ == "__main__":
    unittest.main()
