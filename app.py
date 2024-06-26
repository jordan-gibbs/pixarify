import replicate
from openai import OpenAI
import streamlit as st
import base64
import requests
from PIL import Image


# Function to encode the image
def encode_image(image):
    return base64.b64encode(image).decode('utf-8')


# Function to get the image description from GPT-4
def get_image_description(encoded_image, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this person in the image, their age, gender, hair color, facial expression, and other characteristics, ie facial hair, glasses, or piercings. Also specify the vibe of the photo and the setting using the same brief format. Don't output any other punctuation other than commas. Here is an example output for you: young man, neutral face, light beard, glasses, black hair, warehouse, dark, black and white\n Here is another example for you: young girl, smiling face, glasses, blonde hair, colorful background, happy mood\nbe EXTREMELY concise."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error from GPT-4 API: {response.status_code}, {response.text}")
        return None


# Streamlit app
st.title("📸Pixarify")
st.subheader("✨Transform Yourself into a Pixar Character")

# Retrieve the OpenAI API Key and Replicate token from Streamlit secrets
api_key = st.secrets['OPENAI_API_KEY']
rep_token = st.secrets['REPLICATE_API_TOKEN']

# Set the token for the replicate client
rep_client = replicate.Client(api_token=rep_token)

# Image upload
uploaded_file = st.file_uploader("Upload a close-up headshot or portrait of yourself, and become a Pixar-style character!", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = uploaded_file.read()
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Pixarified Image"):
        # Encode the image
        encoded_image = encode_image(image)

        # Get the image description
        with st.spinner("🪄 Pixarifying your image, please wait..."):
            result = get_image_description(encoded_image, api_key)
            if result and "choices" in result and len(result["choices"]) > 0:
                description = result["choices"][0]["message"]["content"]
                # st.write(f"Description: {description}")

                try:
                    # Generate Pixar-style image using Replicate API
                    # output = rep_client.run(
                    #     "adirik/t2i-adapter-sdxl-depth-midas:8a89b0ab59a050244a751b6475d91041a8582ba33692ae6fab65e0c51b700328",
                    #     input={
                    #         "image": uploaded_file,
                    #         "prompt": f"a Pixar character, {description}, 2022 Pixar style",
                    #         "scheduler": "K_EULER_ANCESTRAL",
                    #         "num_samples": 1,
                    #         "random_seed": 1001,
                    #         "guidance_scale": 9,
                    #         "negative_prompt": "graphic, deformed, mutated, ugly, disfigured, photorealistic, photo",
                    #         "num_inference_steps": 100,
                    #         "adapter_conditioning_scale": 0.66,
                    #         "adapter_conditioning_factor": 1
                    #     }
                    # )

                    output = rep_client.run(
                        "tencentarc/photomaker-style:467d062309da518648ba89d226490e02b8ed09b5abc15026e54e31c5a8cd0769",
                        input={
                            "prompt": f"a Pixar character, {description} img, 3d CGI, art by Pixar, half-body, screenshot from animation",
                            "num_steps": 75,
                            "style_name": "(No style)",
                            "input_image": uploaded_file,
                            "num_outputs": 1,
                            "guidance_scale": 5,
                            "negative_prompt": "realistic, photo-realistic, worst quality, greyscale, bad anatomy, bad hands, error, text",
                            "style_strength_ratio": 35
                        }
                    )

                    if output and len(output) > 0:
                        # Extract the image URL from the output
                        pixarified_image_url = output[0]
                        st.image(pixarified_image_url, caption='Pixarified Image')

                        # Try again button
                        if st.button("Try Again with New Image"):
                            st.experimental_rerun()
                    else:
                        st.error("Failed to generate Pixarified image.")
                except Exception as e:
                    st.error(f"Error generating Pixarified image: {str(e)}")
            else:
                st.error("No description available.")
# else:
#     st.error("Please upload an image file")

