#let conf(
  subject: none,
  title: none,
  subtitle: none,
  authors: (
    (name: "Mikołaj Rosowski", index: 165313),
  ),
  lang: "pl",
  pagenum: none,
  doc,
) = {
  set document(
    title: title,
    author: authors.map(author => author.name),
  )

  set page(
    paper: "a4",
    margin: 2.5cm,
    numbering: pagenum,
  )

  set par(justify: true)

  set text(
    font: "New Computer Modern",
    size: 10pt,
    lang: lang,
  )

  set align(center)
  v(8%)

  if subject != none {
    text(
      size: 1.25em,
      weight: "medium",
    )[#smallcaps(subject)]
    parbreak()
  }

  text(
    size: 1.6em,
    weight: "semibold",
  )[#smallcaps(title)]

  if subtitle != none {
    parbreak()
    text(
      size: 1.25em,
    )[#subtitle]
  }

  parbreak()

  let count = authors.len()
  let ncols = calc.min(count, 3)
  grid(
    columns: (1fr,) * ncols,
    row-gutter: 2.3em,
    ..authors.map(author => text(size: 1.1em)[
      #author.name \
      #author.index
    ]),
  )

  v(3%)

  set align(start + top)

  show heading: it => {
    it
    v(0.4em)
  }

  show raw.where(block: false): box.with(
    fill: luma(240),
    inset: (x: 0.3em, y: 0pt),
    outset: (y: 0.3em),
    radius: 10%,
  )

  doc
}

#let code(content, breakable: false) = {
  set text(size: 0.9em)
  set raw(block: true)
  set align(start)
  block(
    breakable: breakable,
    width: 100%,
    fill: luma(240),
    radius: 0.25em,
    inset: 0.8em,
  )[#content]
}
