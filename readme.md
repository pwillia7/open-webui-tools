# open-webui-tools

> **NOTE:** The previous *_template.py files are now deprecated and broken due to an update with Open-WebUI. Use the two new files in this repository for Txt2Img and Img2Img functionalities.

**open-webui-tools** is a repository that provides tools for integrating ComfyUI workflows with Open-WebUI. These tools let you generate images from text prompts (Txt2Img) or enhance existing images (Img2Img) using customizable, JSON-defined ComfyUI workflows. All configuration settings (such as API key, server address, and workflow URL) are now administered via the admin-controlled Valves.

## 🛠️ Tools

- **Txt2Img Tool:** Generate images from text prompts using a customizable ComfyUI workflow.
- **Img2Img Tool:** Enhance or modify existing images with additional instructions.

## 📦 Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/pwillia7/open-webui-tools.git

   ```
2. **Navigate to the Repository Directory:**
```bash
cd open-webui-tools
```
## ⚙️ Setup
### 1\. Add Tools to Open-WebUI

1.  **Open Open-WebUI:**
    
    -   Launch Open-WebUI in your web browser.
2.  **Navigate to Workspace:**
    
    -   Click on the **Workspace** tab in the Open-WebUI interface.
3.  **Access Tools:**
    
    -   Select the **Tools** section within the Workspace.
4.  **Create a New Tool:**
    
    -   Click on **Create New Tool**.
    -   **Name Your Tool:** For example, "Txt2Img Generator" or "Img2Img Enhancer".
    -   **Paste the Tool Code:**
        -   Open the corresponding Python file from this repository (e.g., `txt2img.py` or `img2img.py`).
        -   Copy the entire file content and paste it into the tool creation interface in Open-WebUI.
    -   **Save the Tool:** Click **Save** to add it to your workspace.

### 2\. Customize Workflow Templates

Both tools contain a `workflow_template` field that is populated by fetching a JSON workflow from a URL. This URL should point to a gist (or another publicly accessible location) that contains your ComfyUI workflow JSON.

#### Txt2Img Tool

-   **Placeholder:** `%%PROMPT%%`
-   **Setup Steps:**
    -   Create or update a gist that contains your ComfyUI workflow JSON.
    -   Make sure your workflow JSON includes the `%%PROMPT%%` placeholder where the text prompt will be injected.
    -   In the tool’s admin settings, set the **Workflow\_URL** to your gist URL.
- [Example Flux Workflow](https://gist.githubusercontent.com/pwillia7/9fe756338c7d35eba130c68408b705f4/raw/4a429e1ede948e02e405e3a046b2eb85546f1c0f/fluxgen)

#### Img2Img Tool

-   **Placeholders:** `%%B64IMAGE%%` (for the input image) and optionally `%%PROMPT%%` (if additional text instructions are used).
-   **Setup Steps:**
    -   Create or update a gist with your ComfyUI workflow JSON.
    -   Ensure the JSON includes the `%%B64IMAGE%%` placeholder (and `%%PROMPT%%` if desired).
    -   In the tool’s admin settings, set the **Workflow\_URL** to point to this gist.
- [Example Enhance! Workflow](https://gist.githubusercontent.com/pwillia7/38bb9fb1da204407339ebe33e66caa35/raw/d0368b32693bb946f9d6a51a7dcec2452138b925/enhance_api_2025.json)

### 3\. Configure API Key and Server Address

Both tools require you to set an API key and the ComfyUI server address. These settings are defined in the admin-only **Valves**. To configure them:

1.  **Access Tool Settings:**
    -   In your Open-WebUI workspace, click the gear icon next to the tool.
2.  **Enter Settings:**
    -   **Api\_Key:** Enter your ComfyUI API key.
    -   **ComfyUI\_Server:** Enter your ComfyUI server address (without the protocol, e.g., `ptkwilliams.ddns.net:8443`).
    -   **Workflow\_URL:** Enter the URL of your gist containing the workflow JSON.
    -   **debug\_mode:** Optionally enable debug mode for extra logging.
3.  **Save Your Settings.**
## 🚀 Usage
### Txt2Img Tool

Invoke the Txt2Img tool with your desired text prompt. The tool replaces the `%%PROMPT%%` placeholder in your JSON workflow with your prompt, submits it to ComfyUI, and displays the generated images within Open-WebUI.

### Img2Img Tool

For image enhancement, provide the URL of the source image along with an optional text prompt. The tool replaces `%%B64IMAGE%%` (and `%%PROMPT%%` if applicable) in your workflow JSON, submits the request, and retrieves the enhanced images.