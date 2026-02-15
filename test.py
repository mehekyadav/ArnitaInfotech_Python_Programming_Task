import unittest
from resume_screening import preprocess_text

class TestResumeScreening(unittest.TestCase):

    def test_preprocess_text_removes_punctuation(self):
        text = "Hello, World!"
        result = preprocess_text(text)
        self.assertEqual(result, "hello world")

    def test_preprocess_text_removes_stopwords(self):
        text = "This is a resume screening system"
        result = preprocess_text(text)
        self.assertEqual(result, "resume screening system")

    def test_preprocess_text_lowercase(self):
        text = "Python is Great!!! Machine Learning is Powerful."
        result = preprocess_text(text)
        self.assertEqual(result, "python great machine learning powerful")

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestResumeScreening)
    runner = unittest.TextTestRunner()

    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n✅ All Test Cases Passed Successfully! Resume Screening System Working Fine.")
    else:
        print("\n❌ Some Test Cases Failed! Please check errors.")
