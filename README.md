# bio.io

A comprehensive collection of plugins and skills for various platforms and services.

## Quick Start

### Installation

You can install bio.io using our installation script:

```bash
curl -fsSL https://raw.githubusercontent.com/princess38827/bio.io/main/install.sh | bash
```

Or download and run it manually:

```bash
wget https://raw.githubusercontent.com/princess38827/bio.io/main/install.sh
chmod +x install.sh
./install.sh
```

### Manual Installation

Alternatively, you can clone the repository directly:

```bash
git clone https://github.com/princess38827/bio.io.git
cd bio.io
```

## Available Plugins

bio.io contains **180+ plugins** covering a wide range of services and platforms, including:

- **Development Tools**: GitHub, GitLab, Linear, Jira
- **Communication**: Slack, Teams, Zoom, Outlook
- **Cloud Platforms**: AWS, Azure, GCP, Vercel, Netlify
- **Databases**: Supabase, MongoDB, PostgreSQL
- **AI/ML**: OpenAI, Hugging Face, Replicate
- **Analytics**: Mixpanel, Amplitude, PostHog
- **CRM**: HubSpot, Salesforce, Pipedrive
- **And many more...**

To see all available plugins:

```bash
ls plugins/
```

## Plugin Structure

Each plugin contains:
- `SKILL.md` - Main documentation and usage instructions
- `references/` - Additional reference materials and examples
- `scripts/` - Helper scripts (where applicable)

## Usage

Navigate to any plugin directory to explore its capabilities:

```bash
cd plugins/<plugin-name>
cat SKILL.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See individual plugin directories for license information.
