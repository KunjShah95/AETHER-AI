---
description: How to publish the Nexus AI VS Code extension to the marketplace
---

# Publishing the Nexus AI VS Code Extension

## Prerequisites

1. **Azure DevOps Personal Access Token (PAT)**
   - Go to https://dev.azure.com/
   - Create a new organization (if you don't have one)
   - Create a Personal Access Token with "Marketplace (Manage)" scope
   - Save the token securely

2. **Publisher Account**
   - Go to https://marketplace.visualstudio.com/manage
   - Create a publisher with ID: `kunjshah95` (matching package.json)

## Steps to Publish

// turbo
1. Navigate to the extension folder:
```bash
cd c:\NEXUS-AI.io\extension
```

// turbo
2. Install dependencies (if not already done):
```bash
npm install
```

// turbo
3. Compile the TypeScript:
```bash
npm run compile
```

// turbo
4. Package the extension:
```bash
vsce package
```

5. Login with your Personal Access Token:
```bash
vsce login kunjshah95
```
(Enter your PAT when prompted)

6. Publish the extension:
```bash
vsce publish
```

## Alternative: Manual Upload

If CLI publishing fails, you can manually upload:

1. Go to https://marketplace.visualstudio.com/manage
2. Click on your publisher
3. Click "New Extension" → "Visual Studio Code"
4. Upload the `.vsix` file from `c:\NEXUS-AI.io\extension\nexus-ai-terminal-extension-0.1.0.vsix`

## Version Bumping

To publish a new version:

// turbo
```bash
vsce publish minor  # for feature releases (0.1.0 → 0.2.0)
vsce publish patch  # for bug fixes (0.1.0 → 0.1.1)
vsce publish major  # for breaking changes (0.1.0 → 1.0.0)
```

## Testing Locally Before Publishing

// turbo
```bash
code --install-extension nexus-ai-terminal-extension-0.1.0.vsix
```

## Unpublishing (if needed)

```bash
vsce unpublish kunjshah95.nexus-ai-terminal-extension
```
