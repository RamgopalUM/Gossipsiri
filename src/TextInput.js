import React, { Component } from 'react'
import TextareaAutosize from "react-textarea-autosize";

class TextInput extends Component {
  constructor(props) {
    super(props);
    this.state = {
      inputText: ''
    };
    this.handleChange = this.handleChange.bind(this);
  }

  handleChange(event) {
    this.setState({inputText: event.target.value});
  }

  render() {
    const { inputText } = this.state;
    return (
      <TextareaAutosize class="textInput" style={{width: '100%', resize:'none', fontSize:'18px'}} minRows={1}
                        maxRows={4} id='ins' placeholder="Type here..." value={inputText}
                        hidden={this.props.speechMode} onChange={this.handleChange}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            this.props.onTextInput(inputText);
                            this.setState({
                              inputText: ''
                            })
                          }
                        }}
      />
    )
  }
}

export default TextInput
