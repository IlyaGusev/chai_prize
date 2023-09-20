import sys
import chai_guanaco as chai

submission_id = sys.argv[1]
model_feedback = chai.get_feedback(submission_id)
model_feedback.sample()

