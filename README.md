<div align="center">
<img src="./assets/logo.svg">

[![Unit Testing](https://github.com/rentruewang/inversql/actions/workflows/unittest.yaml/badge.svg)](https://github.com/rentruewang/inversql/actions/workflows/unittest.yaml)
[![Pre Commit Checks](https://github.com/rentruewang/inversql/actions/workflows/precommit.yaml/badge.svg)](https://github.com/rentruewang/inversql/actions/workflows/precommit.yaml)
[![Publish](https://github.com/rentruewang/inversql/actions/workflows/release.yaml/badge.svg)](https://github.com/rentruewang/inversql/actions/workflows/release.yaml)

![PyPI](https://img.shields.io/pypi/v/inversql)
![MIT](https://img.shields.io/badge/license-MIT-blue)
[![](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)][demo]

<a id="inversql"></a> <h1> 🔄 InverSQL </h1>
</div>

## Generate SQL that match a set of records by decomposing decision trees

1. User doesn't want to write SQL.
2. User uploads CSV to [inversql streamlit app][demo]
3. User selects cells (that will be selected by the SQL).
4. We **overfit** a `scikit-learn` binary decision tree on the data.
5. We decompose the tree, convert to boolean logic (explainable AI part).
6. We simplify the logic with `sympy`.
7. Generate SQL from previous steps (joins to JOIN and boolean to WHERE).
8. User sees the SQL.
9. User is happy.

<a id="demo-gif"></a>
<table>
  <tr>
    <th width="64%">🎬 Demo in a GIF</th>
    <th width="36%">🏛️ Architecture diagram</th>
  </tr>
  <tr>
    <td align="center">
        <img src="./assets/quick-demo.gif" alt="Quick Demo" width="100%" />
      <figcaption>
        Link to
        <a href="https://inversql.streamlit.app">
          live demo site
        </a>
        here.
      </figcation>
    </td>
    <td align="center">
      <img src="./assets/architecture.svg" alt="Architecture Diagram" width="100%" />
    </td>
  </tr>
</table>

### 🏎️ Performance

For each individual SQL query candidate (the shortest one is displayed in the UI), we need to retrain a new decision tree.

But...
The decision tree fitting is honestly fast, don't worry about this.

## 🌟 Give us a star!

That's pretty much it!

If you have read this far, please consider giving me a star (⭐) or a fork (🍴).

This will keep my motivation going!

Or if you have too much cash at hand: [![BuyMeACoffee](https://raw.githubusercontent.com/pachadotdev/buymeacoffee-badges/main/bmc-yellow.svg)](https://www.buymeacoffee.com/rentruewang)

> If you **REALLY** like my work, nowadays I'm working on [`aioway`](https://github.com/rentruewang/aioway), it's an optimizing compiler approach to deep learning, check it out!

### 👨‍👨‍👦‍👦 Contributors

<a href="https://github.com/rentruewang/inversql/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=rentruewang/inversql" />
</a>

Contribution welcome!

To contribute, refer to [CONTRIBUTING.md](./CONTRIBUTING.md),
and our [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).

### 🎨 Inspiration.

Inspired by [regexgen](https://github.com/devongovett/regexgen#how-does-it-work)'s process. Instead of regex we do SQL. Instead of selecting text we do select records. Decision tree is my inspiration tho.

[demo]: https://inversql.streamlit.app
