# Contributing to Computer Genie Dashboard

Thank you for your interest in contributing to Computer Genie Dashboard! We welcome contributions from the community and are excited to see what you'll bring to the project.

## 🚀 Getting Started

### Prerequisites

- Node.js >= 18.0.0
- npm >= 9.0.0 or yarn >= 1.22.0
- Git
- A GitHub account

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/computer-genie-dashboard.git
   cd computer-genie-dashboard
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/originalowner/computer-genie-dashboard.git
   ```
4. **Install dependencies**:
   ```bash
   npm install
   ```
5. **Start the development server**:
   ```bash
   npm run dev
   ```

## 🔄 Development Workflow

### Creating a Feature Branch

1. **Sync with upstream**:
   ```bash
   git checkout main
   git pull upstream main
   ```
2. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Making Changes

1. **Write your code** following our coding standards
2. **Add tests** for new functionality
3. **Update documentation** if needed
4. **Test your changes**:
   ```bash
   npm run test
   npm run lint
   npm run type-check
   ```

### Submitting Changes

1. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add amazing new feature"
   ```
2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
3. **Create a Pull Request** on GitHub

## 📝 Coding Standards

### TypeScript

- Use TypeScript for all new code
- Define proper interfaces and types
- Avoid `any` type unless absolutely necessary
- Use strict mode settings

### Code Style

- Follow the ESLint configuration
- Use Prettier for code formatting
- Use meaningful variable and function names
- Write self-documenting code with clear comments

### Component Guidelines

- Use functional components with hooks
- Implement proper error boundaries
- Follow the single responsibility principle
- Use TypeScript interfaces for props

### File Naming

- Use PascalCase for component files: `MyComponent.tsx`
- Use camelCase for utility files: `myUtility.ts`
- Use kebab-case for directories: `my-feature/`

## 🧪 Testing

### Writing Tests

- Write unit tests for all new functions and components
- Use React Testing Library for component tests
- Aim for at least 80% code coverage
- Test both happy paths and error cases

### Running Tests

```bash
# Run all tests
npm run test

# Run tests in watch mode
npm run test:watch

# Generate coverage report
npm run test:coverage
```

## 📚 Documentation

### Code Documentation

- Document all public APIs
- Use JSDoc comments for functions and classes
- Include examples in documentation
- Keep README.md up to date

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

Examples:
```
feat: add credential management system
fix: resolve workflow validation issue
docs: update API documentation
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear description** of the issue
2. **Steps to reproduce** the bug
3. **Expected behavior**
4. **Actual behavior**
5. **Environment details** (OS, browser, Node.js version)
6. **Screenshots** if applicable

Use our bug report template when creating issues.

## ✨ Feature Requests

For feature requests, please provide:

1. **Clear description** of the feature
2. **Use case** and motivation
3. **Proposed implementation** (if you have ideas)
4. **Alternatives considered**

## 🔍 Code Review Process

### For Contributors

- Ensure your PR has a clear description
- Link related issues
- Keep PRs focused and small
- Respond to feedback promptly
- Update your branch if needed

### For Reviewers

- Be constructive and respectful
- Focus on code quality and maintainability
- Test the changes locally
- Approve when ready or request changes

## 🏷️ Release Process

1. **Version bumping** follows semantic versioning
2. **Changelog** is automatically generated
3. **Releases** are created from the main branch
4. **Tags** are created for each release

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn and grow

### Communication

- Use GitHub issues for bug reports and feature requests
- Use GitHub discussions for questions and ideas
- Be patient and helpful in responses
- Search existing issues before creating new ones

## 🎯 Areas for Contribution

We especially welcome contributions in these areas:

- **New node types** for workflow automation
- **UI/UX improvements** and accessibility
- **Performance optimizations**
- **Documentation** and examples
- **Testing** and quality assurance
- **Internationalization** (i18n)

## 📞 Getting Help

If you need help:

1. Check the [documentation](README.md)
2. Search [existing issues](https://github.com/yourusername/computer-genie-dashboard/issues)
3. Create a new issue with the "question" label
4. Join our [Discord community](https://discord.gg/computer-genie)

## 🙏 Recognition

Contributors will be:

- Listed in the project's contributors section
- Mentioned in release notes for significant contributions
- Invited to join the core team for outstanding contributions

Thank you for contributing to Computer Genie Dashboard! 🚀