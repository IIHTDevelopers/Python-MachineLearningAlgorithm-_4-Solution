import unittest
from test.TestUtils import TestUtils
import pandas as pd
import numpy as np
import io
import sys
import os
import joblib
from employee import load_and_prepare_data, hypothesis_demo, sigmoid_demo, cost_function, train_and_evaluate
from student import load_and_preprocess, show_key_stats, prepare_data, train_and_save_model, evaluate_model


class TestEmployeeAttrition(unittest.TestCase):
    def setUp(self):
        # Initialize TestUtils object for yaksha assertions
        self.test_obj = TestUtils()

    def test_load_and_prepare_data(self):
        """
        Test case for load_and_prepare_data() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function
            df = load_and_prepare_data("employee_attrition.csv")
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if data is loaded correctly
            expected_columns = ['satisfaction_level', 'last_evaluation', 'number_project', 
                               'average_monthly_hours', 'time_spend_company', 'work_accident', 
                               'left', 'promotion_last_5years', 'department', 'salary_level']
            
            if (isinstance(df, pd.DataFrame) and 
                all(col in df.columns for col in df.columns) and
                "✅ Dataset loaded and preprocessed." in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestLoadAndPrepareData", True, "functional")
                print("TestLoadAndPrepareData = Passed")
            else:
                self.test_obj.yakshaAssert("TestLoadAndPrepareData", False, "functional")
                print("TestLoadAndPrepareData = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestLoadAndPrepareData", False, "functional")
            print(f"TestLoadAndPrepareData = Failed | Exception: {e}")

    def test_sigmoid_demo(self):
        """
        Test case for sigmoid_demo() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function
            sigmoid_demo()
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if sigmoid calculation is correct
            expected_value = 0.8808  # sigmoid(2.0) ≈ 0.8808
            
            if "Sigmoid(2.0) = 0.8808" in captured_output.getvalue():
                self.test_obj.yakshaAssert("TestSigmoidDemo", True, "functional")
                print("TestSigmoidDemo = Passed")
            else:
                self.test_obj.yakshaAssert("TestSigmoidDemo", False, "functional")
                print("TestSigmoidDemo = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestSigmoidDemo", False, "functional")
            print(f"TestSigmoidDemo = Failed | Exception: {e}")

    def test_cost_function(self):
        """
        Test case for cost_function() function.
        """
        try:
            # Test with simple values
            y_true = np.array([0, 1, 0, 1])
            y_pred = np.array([0.1, 0.9, 0.2, 0.8])
            
            # Calculate cost
            cost = cost_function(y_true, y_pred)
            
            # Expected cost should be close to -log(0.9) - log(0.9) - log(0.8) - log(0.8) / 4
            expected_cost = 0.1643  # Approximate value
            
            if abs(cost - expected_cost) < 0.1:  # Allow some numerical difference
                self.test_obj.yakshaAssert("TestCostFunction", True, "functional")
                print("TestCostFunction = Passed")
            else:
                self.test_obj.yakshaAssert("TestCostFunction", False, "functional")
                print("TestCostFunction = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestCostFunction", False, "functional")
            print(f"TestCostFunction = Failed | Exception: {e}")

    def test_hypothesis_demo(self):
        """
        Test case for hypothesis_demo() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function
            hypothesis_demo()
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if output contains expected text
            if ("Hypothesis h(x) = sigmoid(w·x + b)" in captured_output.getvalue() and
                "Probability that employee will leave" in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestHypothesisDemo", True, "functional")
                print("TestHypothesisDemo = Passed")
            else:
                self.test_obj.yakshaAssert("TestHypothesisDemo", False, "functional")
                print("TestHypothesisDemo = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestHypothesisDemo", False, "functional")
            print(f"TestHypothesisDemo = Failed | Exception: {e}")

    def test_train_and_evaluate(self):
        """
        Test case for train_and_evaluate() function.
        """
        try:
            # Prepare minimal test data
            df = load_and_prepare_data("employee_attrition.csv")
            X = df.drop(columns=['left'])
            y = df['left']
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function with a test model path
            test_model_path = "test_employee_model.pkl"
            train_and_evaluate(X_train, y_train, X_test, y_test, path=test_model_path)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if model was created and output contains expected text
            if (os.path.exists(test_model_path) and 
                "Model trained and saved" in captured_output.getvalue() and
                "Log Loss (Custom Cost)" in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestTrainAndEvaluate", True, "functional")
                print("TestTrainAndEvaluate = Passed")
            else:
                self.test_obj.yakshaAssert("TestTrainAndEvaluate", False, "functional")
                print("TestTrainAndEvaluate = Failed")
                
            # Clean up test model file
            if os.path.exists(test_model_path):
                os.remove(test_model_path)
        except Exception as e:
            self.test_obj.yakshaAssert("TestTrainAndEvaluate", False, "functional")
            print(f"TestTrainAndEvaluate = Failed | Exception: {e}")


class TestStudentScores(unittest.TestCase):
    def setUp(self):
        # Initialize TestUtils object for yaksha assertions
        self.test_obj = TestUtils()
        
        # Prepare test data for student.py
        self.features = ['hours_studied', 'previous_score', 'assignments_completed']
        self.target = 'final_score'

    def test_load_and_preprocess(self):
        """
        Test case for load_and_preprocess() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function
            df = load_and_preprocess("student_scores.csv")
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if data is loaded correctly
            expected_columns = ['hours_studied', 'previous_score', 'assignments_completed', 'final_score']
            
            if (isinstance(df, pd.DataFrame) and 
                all(col in df.columns for col in expected_columns) and
                "Student data loaded and cleaned." in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestLoadAndPreprocess", True, "functional")
                print("TestLoadAndPreprocess = Passed")
            else:
                self.test_obj.yakshaAssert("TestLoadAndPreprocess", False, "functional")
                print("TestLoadAndPreprocess = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestLoadAndPreprocess", False, "functional")
            print(f"TestLoadAndPreprocess = Failed | Exception: {e}")

    def test_show_key_stats(self):
        """
        Test case for show_key_stats() function.
        """
        try:
            # Load data for testing
            df = load_and_preprocess("student_scores.csv")
            
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function
            show_key_stats(df)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if output contains expected text
            if ("Standard Deviation of Study Hours" in captured_output.getvalue() and
                "Max Previous Score" in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestShowKeyStats", True, "functional")
                print("TestShowKeyStats = Passed")
            else:
                self.test_obj.yakshaAssert("TestShowKeyStats", False, "functional")
                print("TestShowKeyStats = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestShowKeyStats", False, "functional")
            print(f"TestShowKeyStats = Failed | Exception: {e}")

    def test_prepare_data(self):
        """
        Test case for prepare_data() function.
        """
        try:
            # Load data for testing
            df = load_and_preprocess("student_scores.csv")
            
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function
            X_train, X_test, y_train, y_test, scaler = prepare_data(df, self.features, self.target)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if data is prepared correctly
            if (isinstance(X_train, np.ndarray) and 
                isinstance(X_test, np.ndarray) and
                isinstance(y_train, pd.Series) and
                isinstance(y_test, pd.Series) and
                "Data prepared and split." in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestPrepareData", True, "functional")
                print("TestPrepareData = Passed")
            else:
                self.test_obj.yakshaAssert("TestPrepareData", False, "functional")
                print("TestPrepareData = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestPrepareData", False, "functional")
            print(f"TestPrepareData = Failed | Exception: {e}")

    def test_train_and_save_model(self):
        """
        Test case for train_and_save_model() function.
        """
        try:
            # Prepare data for testing
            df = load_and_preprocess("student_scores.csv")
            X_train, X_test, y_train, y_test, scaler = prepare_data(df, self.features, self.target)
            
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function with a test model path
            test_model_path = "test_student_model.pkl"
            model = train_and_save_model(X_train, y_train, model_path=test_model_path)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if model was created and output contains expected text
            if (os.path.exists(test_model_path) and 
                "Model trained and saved" in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestTrainAndSaveModel", True, "functional")
                print("TestTrainAndSaveModel = Passed")
            else:
                self.test_obj.yakshaAssert("TestTrainAndSaveModel", False, "functional")
                print("TestTrainAndSaveModel = Failed")
                
            # Clean up test model file
            if os.path.exists(test_model_path):
                os.remove(test_model_path)
        except Exception as e:
            self.test_obj.yakshaAssert("TestTrainAndSaveModel", False, "functional")
            print(f"TestTrainAndSaveModel = Failed | Exception: {e}")

    def test_evaluate_model(self):
        """
        Test case for evaluate_model() function.
        """
        try:
            # Prepare data and model for testing
            df = load_and_preprocess("student_scores.csv")
            X_train, X_test, y_train, y_test, scaler = prepare_data(df, self.features, self.target)
            model = train_and_save_model(X_train, y_train, model_path="test_student_model.pkl")
            
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function
            evaluate_model(model, X_test, y_test)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if output contains expected text
            if ("Mean Squared Error" in captured_output.getvalue() and
                "Sample Predictions" in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestEvaluateModel", True, "functional")
                print("TestEvaluateModel = Passed")
            else:
                self.test_obj.yakshaAssert("TestEvaluateModel", False, "functional")
                print("TestEvaluateModel = Failed")
                
            # Clean up test model file
            if os.path.exists("test_student_model.pkl"):
                os.remove("test_student_model.pkl")
        except Exception as e:
            self.test_obj.yakshaAssert("TestEvaluateModel", False, "functional")
            print(f"TestEvaluateModel = Failed | Exception: {e}")


if __name__ == '__main__':
    unittest.main()
