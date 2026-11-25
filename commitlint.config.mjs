// commitlint: Conventional Commits ("type(scope): summary"), kept deliberately loose.
// Levels: 0 = off, 1 = warning (commit goes through), 2 = error (commit is rejected).
export default {
  extends: ["@commitlint/config-conventional"],

  // Peanuts creates commits by itself and does not bypass hooks:
  //   peanuts(<theme>): agent update
  //   peanuts(<theme>): manual merge commit
  //   peanuts: merge <theme> (<branch>)
  // <theme> is free text (up to 120 characters), so skip these entirely
  // instead of trying to make them fit the rules.
  ignores: [(message) => /^peanuts(\(|: )/.test(message)],

  // Keep commitlint's built-in ignores: Merge ..., Revert ..., fixup!, squash!, etc.
  defaultIgnores: true,

  rules: {
    // Japanese subjects and proper nouns make case rules misfire.
    "subject-case": [0],
    // URLs, logs and Co-Authored-By trailers are often long.
    "body-max-line-length": [0],
    "footer-max-line-length": [0],
    // Nudge only; do not block.
    "header-max-length": [1, "always", 100],
    "subject-full-stop": [1, "never", "."],
  },
};
