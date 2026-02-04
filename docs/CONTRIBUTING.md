# Contributing to Glitch Mirror

Thank you for your interest in contributing to Glitch Mirror!

## How to Contribute

### Reporting Bugs
1. Check if the bug has already been reported in Issues
2. If not, create a new issue with:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Your system info (OS, Python version, etc.)

### Suggesting Features
1. Open an issue with the "feature request" label
2. Describe the feature and its use case
3. Include mockups or examples if possible

### Code Contributions
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Code Style
- Follow PEP 8 for Python code
- Use descriptive variable and function names
- Add comments for complex logic
- Update documentation if needed

## Adding New Effects
New effects should be added to `effects.py` or `object_detection.py`:

1. Add the effect function
2. Add to `AVAILABLE_EFFECTS` list
3. Add UI controls in `detection_panel.py` if needed
4. Update README with effect description

## Testing
Before submitting:
- Test with webcam input
- Test with video file input
- Test with static image input
- Verify effects render correctly in output

## Questions?
Open an issue with the "question" label.
