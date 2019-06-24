remark.macros.scale = function (percentage) {
  var url = this;
  return '<img src="' + url + '" style="width: ' + percentage + '" />';
};

remark.macros.vspace = function (percentage) {
  return `<div style="height: ${percentage}%"></div>`
};

remark.macros.centerScale = function (percentage) {
  var url = this;
  return `<div style="text-align:center;">
            <img src='${url}' style=width: ${percentage}%/>
          </div>`;
};
