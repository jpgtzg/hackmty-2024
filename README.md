# AgriSense 🌱

A tool with AI for plant detection and care could leverage technologies such as computer vision and machine learning to identify and monitor plant health.

![Copy of AgriSense](https://github.com/user-attachments/assets/25abb697-e59e-4d5b-8a42-028e84674290)

## Problematic 🔗

The agricultural sector is one of the most critical industries we rely on, as it forms the basis of our ability to feed the global population. However, the processes behind it are often complex and costly, especially considering that more than **70% of the world's freshwater is allocated to agriculture**. This heavy water usage exacerbates the ongoing global water crises and contributes directly to climate change. In the end, this value chain cna get really expensive and consuming in the long term. According to the INEGI, at least in Mexico, in 2023 they lost 305,000 USD in all this value chain. This value includes:

- Costs of damage
- Production losses
- Fertilizers
- Plagues and plant deseases
- Transportation

## Our solution 💡

We have developed an advanced farm management app that integrates a trained YOLO object detection model to monitor plant health. The model generates a JSON file, which is processed by a trained OpenAI LLM to provide actionable insights, recommendations, and instructions. Additionally, the app manages IoT devices, such as water pipes, to automate and optimize the irrigation process, ensuring efficient water usage and crop management.

Our app not only manages farm operations and plant health using AI, but it also evaluates **the quality of fruits and vegetables** when they are ready for packaging. The app can determine which client the product should be sold to based on quality, or provide suggestions for handling products that don’t meet the required standards.

## Tech Stack 💻

### [Back-end](https://github.com/jpgtzg/hackmty-2024)
- FRIDA Framework (provided by SoftTek)
- OpenAI API
- AWS API Gateway
- Flask
- OpenVINO
- OpenCV
