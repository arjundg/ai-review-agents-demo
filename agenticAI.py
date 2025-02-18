import streamlit as st
from autogen import AssistantAgent  # Assuming autogen library is installed and AssistantAgent is available.

def reflection_message(recipient, messages, sender, config):
    return f'''Review the following content. 
            \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}'''

# Streamlit App Interface
st.title("Text Review and Analysis Tool")

# Text input for the user
text_input = st.text_area("Enter the text you want to analyze", height=200)

# Checkbox options for the types of review
st.write("Select the type(s) of analysis required:")
seo_check = st.checkbox("SEO Analysis")
legal_check = st.checkbox("Legal Analysis")
ethics_check = st.checkbox("Ethics Review")

# Input for OpenAI API Key
api_key = st.text_input("Provide your OpenAI API Key", type="password")

# Submit button
if st.button("Submit"):
    if not api_key:
        st.error("Please provide your OpenAI API Key.")
    elif not text_input:
        st.error("Please enter text for analysis.")
    elif not (seo_check or legal_check or ethics_check):
        st.error("Please select at least one type of analysis.")
    else:
        llm_config = {"config_list": [{
            "model": "gpt-3.5-turbo"
        }]}

        writer = AssistantAgent(
            name="Writer",
            system_message="You are a writer. You have written this post. Repeat this post to the reviewers. ",
            llm_config=llm_config,
        )
        critic = AssistantAgent(
            name="Critic",
            is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
            llm_config=llm_config,
            system_message="You are a critic. You review the work of "
                        "the writer and provide constructive "
                        "feedback to help improve the quality of the content.",
        )

        # Define agents and tasks
        # Create agents for each selected task
        review_chats = []
        if seo_check:
            agent = AssistantAgent(
                name="SEO_Reviewer",
                is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
                llm_config=llm_config,
                system_message="You are an SEO reviewer, known for "
                    "your ability to optimize content for search engines, "
                    "ensuring that it ranks well and attracts organic traffic. " 
                    "Make sure your suggestion is concise (within 3 bullet points), "
                    "concrete and to the point. "
                    "Begin the review by stating your role.",
            )
            review_chats.append({
                "recipient": agent,
                "message": reflection_message,
                "summary_method": "reflection_with_llm",
                "summary_args": {"summary_prompt" :
                    "Return review into as JSON object only:"
                    "{'Reviewer': '', 'Review': ''}. Here Reviewer should be your role",},
                "max_turns": 1
                })
        if legal_check:
            agent = AssistantAgent(
                name="Legal_Reviewer",
                is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
                llm_config=llm_config,
                system_message="You are a legal reviewer, known for "
                    "your ability to ensure that content is legally compliant "
                    "and free from any potential legal issues. "
                    "Make sure your suggestion is concise (within 3 bullet points), "
                    "concrete and to the point. "
                    "Begin the review by stating your role.",
            )
            review_chats.append({
                "recipient": agent,
                "message": reflection_message,
                "summary_method": "reflection_with_llm",
                "summary_args": {"summary_prompt" :
                    "Return review into as JSON object only:"
                    "{'Reviewer': '', 'Review': ''}. Here Reviewer should be your role",},
                "max_turns": 1
                })
        if ethics_check:
            agent = AssistantAgent(
                name="Ethics_Reviewer",
                is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
                llm_config=llm_config,
                system_message="You are an ethics reviewer, known for "
                    "your ability to ensure that content is ethically sound "
                    "and free from any potential ethical issues. " 
                    "Make sure your suggestion is concise (within 3 bullet points), "
                    "concrete and to the point. "
                    "Begin the review by stating your role. ",
            )
            review_chats.append({
                "recipient": agent,
                "message": reflection_message,
                "summary_method": "reflection_with_llm",
                "summary_args": {"summary_prompt" :
                    "Return review into as JSON object only:"
                    "{'Reviewer': '', 'Review': ''}. Here Reviewer should be your role",},
                "max_turns": 1
                })
        
        meta_reviewer = AssistantAgent(
                name="Meta_Reviewer",
                is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
                llm_config=llm_config,
                system_message="You are a meta reviewer, you aggragate and review "
                    "the work of other reviewers and give a final suggestion on the content.",
            )
        
        review_chats.append({
            "recipient": meta_reviewer,
            "message": "Aggregate feedback from all reviewers and give final suggestions on the writing.",
            "max_turns": 1
        })
        
        critic.register_nested_chats(
            review_chats,
            trigger=writer,
        )

        res = critic.initiate_chat(
            recipient=writer,
            message=text_input,
            max_turns=2,
            summary_method="last_msg"
        )

        # Show conversation flow
        st.subheader("Agent Conversations:")
        for chat in res.chat_history:
            st.write(chat["name"] +":")
            st.write(chat["content"])
            #st.write(res.chat_history)
        
        # Display the final output
        st.subheader("Final Review after Analysis:")
        st.write(res.summary)