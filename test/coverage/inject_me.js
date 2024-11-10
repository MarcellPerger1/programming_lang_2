// Copyright (c) 2024 Marcell Perger, available under the MIT license, see
// https://github.com/MarcellPerger1/programming_lang_2

// Add fancy colors, works in dark mode, untested in light mode.
// IMPORTANT: do not remove or change the top line! My build scripts that inject
//  colors into the htmlcov depend on it being there (so they don't append this twice)!
function colors_getPercent(ra) {
  let [a, b] =ra.split(' ').map(Number);
  if(b==0) return 100;
  return 100*a / b;
}
function colors_getCls(el) {
  let r = el.dataset.ratio;
  let pc = colors_getPercent(r);
  if (pc >= 90) return 'green';
  if (pc >= 75) return 'orange';
  return 'red';
}
function setColors() {
  document.querySelectorAll('tr td:last-of-type')
    .forEach((el) => el.classList.add(colors_getCls(el)));
}
function linkifyRow(tr) {
  let nameField = tr.children[0]
  let a = nameField.children[0];
  let href = a.href;
  [...tr.children].forEach((el, i) => {  // Copy to avoid problems with iterating while modifying
    console.log(el, i);
    if(i==0) return;
    if(el.querySelector('td > a')) return;  // Already has an <a>
    let aNew = document.createElement('a');
    aNew.href = href;
    aNew.append(...el.childNodes);  // Transfer all children to inner <a>
    el.append(aNew);
  })
}
function linkifyAll() {
  document.querySelectorAll('tr.region').forEach(linkifyRow);
}
addEventListener('load', () => {
  if(document.body.classList.contains("indexfile")) {
    setColors();
    linkifyAll();
  }
});
