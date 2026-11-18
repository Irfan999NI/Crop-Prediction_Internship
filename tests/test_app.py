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
                "Temperature": "20.87974371",
                "Humidity": "82.00274423",
                "pH": "6.502985292",
                "Rainfall": "202.9355362",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"rice", response.data.lower())
        self.assertIn(b"fertility", response.data.lower())
        self.assertIn(b"fertilizer guidance", response.data.lower())
        self.assertIn(b'href="/" class="btn btn-reset">Clear values</a>', response.data)

    def test_out_of_training_range_values_are_rejected(self):
        response = self.client.post(
            "/predict",
            data={
                "Nitrogen": "0",
                "Phosphorus": "0",
                "Potassium": "0",
                "Temperature": "0",
                "Humidity": "0",
                "pH": "0",
                "Rainfall": "0",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"phosphorus must be between", response.data.lower())
        self.assertNotIn(b'class="result-crop"', response.data.lower())

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
