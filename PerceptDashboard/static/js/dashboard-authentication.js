function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
const csrftoken = getCookie('csrftoken');

function validateEmail(mail)
{
  if (/^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/.test(mail)) return true;
  return false;
}

function showNotification(type, from, align, message, delay) {
  if (type == "success")
  {
    icon = "ni ni-like-2"
  }
  else
  {
    icon = ""
  }

  $.notify({
    icon: icon ,
    message: message
  }, {
    type: type,
    timer: delay,
    placement: {
      from: from,
      align: align
    }
  });
}
