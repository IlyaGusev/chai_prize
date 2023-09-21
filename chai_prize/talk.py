import sys
import chai_guanaco as chai

submission_id = sys.argv[1]
chatbot = chai.SubmissionChatbot(submission_id)
#chatbot.chat("vampire_queen", show_model_input=True)
chatbot.chat("nerd_girl", show_model_input=True)
