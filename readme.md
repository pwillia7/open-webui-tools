# open-webui-tools

**open-webui-tools** is a repository that provides tools and templates for integrating ComfyUI workflows with Open-WebUI. It includes both **txt2img** and **img2img** functionalities, enabling users to generate images from text prompts or enhance existing images seamlessly.

## 🛠️ Features

- **Txt2Img Tool Template:** Generate images based on text prompts using customizable ComfyUI workflows.
- **Img2Img Tool Template:** Enhance or modify existing images with user-provided instructions.
- **Easy Integration:** Simple setup process within Open-WebUI's workspace.
- **Customizable Workflows:** Swap out workflow templates to fit your specific needs.
- **Event Emission:** Integrates with chat interfaces to display generated images and status updates.

## 📦 Installation

1. **Clone the Repository:**
   - Download the repository to your local machine using Git:
     ```
     git clone https://github.com/pwillia7/open-webui-tools.git
     ```
   
2. **Navigate to the Repository Directory:**
   - Move into the cloned repository folder:
     ```
     cd open-webui-tools
     ```

## ⚙️ Setup

### 1. Add Tools to Open-WebUI

1. **Open Open-WebUI:**
   - Launch your Open-WebUI application in your web browser.

2. **Navigate to Workspace:**
   - Click on the **Workspace** tab in the Open-WebUI interface.

3. **Access Tools:**
   - Click on the **Tools** section within the Workspace.

4. **Create a New Tool:**
   - Click on **Create New Tool**.
   - **Name Your Tool:** Provide a descriptive name (e.g., "Txt2Img Generator" or "Img2Img Enhancer").
   - **Paste the Tool Code:**
     - Open the corresponding tool Python file from the `open-webui-tools` repository (`txt2img_template.py` or `img2img_template.py`).
     - Copy the entire code and paste it into the tool creation interface in Open-WebUI.

5. **Save the Tool:**
   - Click **Save** to add the tool to your workspace.

### 2. Customize Workflow Templates

Both tools include a `workflow_template` section containing JSON for ComfyUI workflows. Customize these templates to fit your specific requirements.

#### Txt2Img Template

- **Placeholder:** `%%PROMPT%%`
- **Customization Steps:**
  - Replace the placeholder JSON in the `workflow_template` with your actual ComfyUI workflow JSON.
  - Ensure the workflow includes the `%%PROMPT%%` placeholder where the text prompt should be injected.

#### Img2Img Template

- **Placeholders:** `%%B64IMAGE%%` and optionally `%%PROMPT%%`
- **Customization Steps:**
  - Replace the placeholder JSON in the `workflow_template` with your actual ComfyUI workflow JSON.
  - Ensure the workflow includes the `%%B64IMAGE%%` placeholder in the `base64image` input node.
  - Optionally, include the `%%PROMPT%%` placeholder if your workflow utilizes text prompts.

### 3. Configure API Key and Server Address

Both tools require an API key and server address for authentication and communication with the ComfyUI server.

1. **Set API Key:**
   - In the Open Webui workspace settings, navigate to tools and press the gear icon next to your tool. 
   - Enter your ComfyUI API key and server address without protocol. 
   
## 🚀 Usage

### Txt2Img Tool

Use the Txt2Img tool to generate images based on text prompts. Simply invoke the tool within Open-WebUI with your desired prompt, and the tool will handle the image generation and display.

### Img2Img Tool

Use the Img2Img tool to enhance or modify existing images. Provide the image URL and optional instructions, and the tool will process and display the enhanced images within Open-WebUI.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.
